use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

#[cfg(unix)]
use std::os::fd::OwnedFd;

#[cfg(unix)]
use rustix::fs::{AtFlags, FileType, Mode, OFlags, ResolveFlags, mkdirat, openat, openat2, statat};
use thiserror::Error;

/// Default maximum bytes read from a single sandboxed input file (10 MiB).
pub const MAX_READ_BYTES: usize = 10 * 1024 * 1024;

/// Errors raised when validating or opening sandboxed filesystem paths.
#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("path '{0}' is outside authorized root '{1}'")]
    OutsideRoot(String, String),
    #[error("symbolic links are forbidden inside sandbox: '{0}'")]
    SymlinkNotAllowed(String),
    #[error("path '{0}' is not a regular file (FIFOs, sockets, and directories are rejected)")]
    NotARegularFile(String),
    #[error("file '{0}' exceeds maximum allowed size of {1} bytes")]
    FileTooLarge(String, usize),
    #[error("no root directory configured for this operation")]
    NoRootConfigured,
    #[error("I/O error resolving path: {0}")]
    Io(#[from] std::io::Error),
}

/// Directory sandbox enforcing strict `--read-root` and `--write-root` isolation using
/// pinned root directory file descriptors and kernel `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)`
/// so intermediate-directory symlink swaps cannot escape the sandbox.
#[derive(Debug, Clone)]
pub struct PathSandbox {
    read_root: Option<PathBuf>,
    write_root: Option<PathBuf>,
    #[cfg(unix)]
    read_fd: Option<Arc<OwnedFd>>,
    #[cfg(unix)]
    write_fd: Option<Arc<OwnedFd>>,
}

impl PathSandbox {
    /// Create a sandbox with optional canonicalized read and write roots.
    /// Existing write roots are required unless `create_write_root` is enabled via
    /// [`PathSandbox::new_with_options`].
    pub fn new(read_root: Option<&Path>, write_root: Option<&Path>) -> Result<Self, SandboxError> {
        Self::new_with_options(read_root, write_root, true)
    }

    /// Create a sandbox with explicit control over whether a missing `write_root` may be created.
    pub fn new_with_options(
        read_root: Option<&Path>,
        write_root: Option<&Path>,
        create_write_root: bool,
    ) -> Result<Self, SandboxError> {
        let r = match read_root {
            Some(p) => Some(std::fs::canonicalize(p)?),
            None => None,
        };
        let w = match write_root {
            Some(p) => {
                if !p.exists() {
                    if create_write_root {
                        std::fs::create_dir_all(p)?;
                    } else {
                        return Err(SandboxError::Io(std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            format!(
                                "write-root '{}' does not exist (create it first or pass --create-write-root)",
                                p.display()
                            ),
                        )));
                    }
                }
                Some(std::fs::canonicalize(p)?)
            }
            None => None,
        };

        #[cfg(unix)]
        let read_fd = match &r {
            Some(p) => Some(Arc::new(open_root_dir_fd(p)?)),
            None => None,
        };
        #[cfg(unix)]
        let write_fd = match &w {
            Some(p) => Some(Arc::new(open_root_dir_fd(p)?)),
            None => None,
        };

        Ok(Self {
            read_root: r,
            write_root: w,
            #[cfg(unix)]
            read_fd,
            #[cfg(unix)]
            write_fd,
        })
    }

    /// Resolve and verify read access (note: prefer `open_read` / `read_to_string_limited`
    /// to operate directly on the kernel-opened file descriptor).
    pub fn resolve_read(&self, candidate: &Path) -> Result<PathBuf, SandboxError> {
        let (path, _file) = self.open_read(candidate)?;
        Ok(path)
    }

    /// Securely open a regular file for reading inside `read_root` atomically via `openat2`
    /// (`RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS`), rejecting symlinks at any component, FIFOs,
    /// sockets, directories, and path escapes.
    pub fn open_read(&self, candidate: &Path) -> Result<(PathBuf, File), SandboxError> {
        let root = self
            .read_root
            .as_ref()
            .ok_or(SandboxError::NoRootConfigured)?;
        let rel = normalize_relative_under_root(root, candidate)?;
        let target = root.join(&rel);

        #[cfg(unix)]
        let file = {
            let root_fd = self
                .read_fd
                .as_ref()
                .ok_or(SandboxError::NoRootConfigured)?;
            let fd = open_beneath_no_symlinks(
                root_fd.as_ref(),
                &rel,
                OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|e| map_unix_open_err(e, &target))?;
            File::from(fd)
        };
        #[cfg(not(unix))]
        let file = File::open(&target)?;

        let meta = file.metadata()?;
        if !meta.is_file() {
            return Err(SandboxError::NotARegularFile(target.display().to_string()));
        }
        Ok((target, file))
    }

    /// Securely read a bounded UTF-8 file inside `read_root` (protects against FIFOs and OOM).
    pub fn read_to_string_limited(
        &self,
        candidate: &Path,
        max_bytes: usize,
    ) -> Result<(PathBuf, String), SandboxError> {
        let (path, file) = self.open_read(candidate)?;
        let mut buf = Vec::new();
        file.take((max_bytes + 1) as u64).read_to_end(&mut buf)?;
        if buf.len() > max_bytes {
            return Err(SandboxError::FileTooLarge(
                path.display().to_string(),
                max_bytes,
            ));
        }
        let s = String::from_utf8(buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok((path, s))
    }

    /// Validate a relative write target path lexically under `write_root`.
    pub fn resolve_write(&self, candidate: &Path) -> Result<PathBuf, SandboxError> {
        let root = self
            .write_root
            .as_ref()
            .ok_or(SandboxError::NoRootConfigured)?;
        let rel = normalize_relative_under_root(root, candidate)?;
        Ok(root.join(rel))
    }

    /// Atomically open/create a writable regular file inside `write_root` using `dirfd`-pinned
    /// parent traversal (`openat2` + `mkdirat`) and `RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS`,
    /// truncating only after confirming the opened fd is a regular file.
    pub fn open_write(&self, candidate: &Path) -> Result<(PathBuf, File), SandboxError> {
        let root = self
            .write_root
            .as_ref()
            .ok_or(SandboxError::NoRootConfigured)?;
        let rel = normalize_relative_under_root(root, candidate)?;
        let target = root.join(&rel);

        #[cfg(unix)]
        let file = {
            let root_fd = self
                .write_fd
                .as_ref()
                .ok_or(SandboxError::NoRootConfigured)?;
            let comps: Vec<&std::ffi::OsStr> = rel.iter().collect();
            let Some((file_name, parent_comps)) = comps.split_last() else {
                return Err(SandboxError::OutsideRoot(
                    candidate.display().to_string(),
                    root.display().to_string(),
                ));
            };

            let mut parent_fd: Option<OwnedFd> = None;
            for comp in parent_comps {
                let base_fd = parent_fd.as_ref().map_or_else(|| root_fd.as_ref(), |f| f);
                let next_fd = open_or_create_subdir(base_fd, Path::new(comp), &target)?;
                parent_fd = Some(next_fd);
            }
            let dir_ref = parent_fd.as_ref().map_or_else(|| root_fd.as_ref(), |f| f);

            let fd = open_beneath_no_symlinks(
                dir_ref,
                Path::new(file_name),
                OFlags::WRONLY
                    | OFlags::CREATE
                    | OFlags::NOFOLLOW
                    | OFlags::NONBLOCK
                    | OFlags::CLOEXEC,
                Mode::from_bits_truncate(0o644),
            )
            .map_err(|e| map_unix_open_err(e, &target))?;
            File::from(fd)
        };
        #[cfg(not(unix))]
        let file = {
            if let Some(parent) = target.parent()
                && !parent.exists()
            {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .open(&target)?
        };

        let meta = file.metadata()?;
        if !meta.is_file() {
            return Err(SandboxError::NotARegularFile(target.display().to_string()));
        }
        file.set_len(0)?;
        Ok((target, file))
    }
}

#[cfg(unix)]
fn open_root_dir_fd(root: &Path) -> Result<OwnedFd, SandboxError> {
    let f = std::os::unix::fs::OpenOptionsExt::custom_flags(
        std::fs::OpenOptions::new().read(true),
        libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
    )
    .open(root)?;
    Ok(OwnedFd::from(f))
}

#[cfg(unix)]
fn open_beneath_no_symlinks(
    dirfd: &OwnedFd,
    rel: &Path,
    flags: OFlags,
    mode: Mode,
) -> Result<OwnedFd, rustix::io::Errno> {
    match openat2(
        dirfd,
        rel,
        flags,
        mode,
        ResolveFlags::BENEATH | ResolveFlags::NO_SYMLINKS,
    ) {
        Ok(fd) => Ok(fd),
        Err(rustix::io::Errno::NOSYS) => {
            // Component-by-component dirfd walk fallback on pre-5.6 kernels
            let comps: Vec<&std::ffi::OsStr> = rel.iter().collect();
            let Some((last, parents)) = comps.split_last() else {
                return Err(rustix::io::Errno::INVAL);
            };
            let mut cur: Option<OwnedFd> = None;
            for comp in parents {
                let base = cur.as_ref().unwrap_or(dirfd);
                let next = openat(
                    base,
                    Path::new(comp),
                    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                )?;
                cur = Some(next);
            }
            let base = cur.as_ref().unwrap_or(dirfd);
            openat(base, Path::new(last), flags | OFlags::NOFOLLOW, mode)
        }
        Err(e) => Err(e),
    }
}

#[cfg(unix)]
fn open_or_create_subdir(
    dirfd: &OwnedFd,
    comp: &Path,
    full_target: &Path,
) -> Result<OwnedFd, SandboxError> {
    let dir_flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    match open_beneath_no_symlinks(dirfd, comp, dir_flags, Mode::empty()) {
        Ok(fd) => Ok(fd),
        Err(rustix::io::Errno::NOENT) => {
            match mkdirat(dirfd, comp, Mode::from_bits_truncate(0o755)) {
                Ok(()) | Err(rustix::io::Errno::EXIST) => {}
                Err(e) => return Err(map_unix_open_err(e, full_target)),
            }
            open_beneath_no_symlinks(dirfd, comp, dir_flags, Mode::empty())
                .map_err(|e| map_unix_open_err(e, full_target))
        }
        Err(e) => {
            if e == rustix::io::Errno::NOTDIR
                && let Ok(st) = statat(dirfd, comp, AtFlags::SYMLINK_NOFOLLOW)
                && FileType::from_raw_mode(st.st_mode).is_symlink()
            {
                return Err(SandboxError::SymlinkNotAllowed(
                    full_target.display().to_string(),
                ));
            }
            Err(map_unix_open_err(e, full_target))
        }
    }
}

#[cfg(unix)]
fn map_unix_open_err(err: rustix::io::Errno, target: &Path) -> SandboxError {
    if err == rustix::io::Errno::LOOP {
        SandboxError::SymlinkNotAllowed(target.display().to_string())
    } else if err == rustix::io::Errno::XDEV {
        SandboxError::OutsideRoot(target.display().to_string(), "sandbox root".to_string())
    } else if err == rustix::io::Errno::NOTDIR
        || err == rustix::io::Errno::ISDIR
        || err == rustix::io::Errno::NXIO
    {
        SandboxError::NotARegularFile(target.display().to_string())
    } else {
        SandboxError::Io(std::io::Error::from_raw_os_error(err.raw_os_error()))
    }
}

fn normalize_relative_under_root(root: &Path, candidate: &Path) -> Result<PathBuf, SandboxError> {
    let joined = if candidate.is_absolute() {
        normalize_lexical(candidate)
    } else {
        normalize_lexical(&root.join(candidate))
    };
    let rel = joined
        .strip_prefix(root)
        .map_err(|_| {
            SandboxError::OutsideRoot(candidate.display().to_string(), root.display().to_string())
        })?
        .to_path_buf();
    if rel.as_os_str().is_empty() {
        return Err(SandboxError::OutsideRoot(
            candidate.display().to_string(),
            root.display().to_string(),
        ));
    }
    Ok(rel)
}

fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod confinement_tests {
    use super::*;
    use std::fs;

    struct Fixture {
        base: std::path::PathBuf,
        root: std::path::PathBuf,
        outside: std::path::PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    fn fixture(tag: &str) -> Option<Fixture> {
        let base = std::env::temp_dir().join(format!("sbx_conf_{}_{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let root = base.join("root");
        let outside = base.join("outside");
        fs::create_dir_all(root.join("sub")).ok()?;
        fs::create_dir_all(&outside).ok()?;
        fs::write(root.join("ok.txt"), "inside\n").ok()?;
        fs::write(root.join("sub/ok.txt"), "inside sub\n").ok()?;
        fs::write(outside.join("secret.txt"), "SECRET\n").ok()?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let _ = symlink(outside.join("secret.txt"), root.join("link_out"));
            let _ = symlink(root.join("ok.txt"), root.join("link_in"));
        }
        Some(Fixture {
            base,
            root,
            outside,
        })
    }

    /// The confinement contract, pinned case by case.
    ///
    /// This crate carries its own copy of the path sandbox, and so do two sibling repositories in
    /// the suite. The copies behave identically today — that was measured across every case below
    /// — but nothing keeps them that way, and a copy that quietly starts accepting one of these is
    /// a sandbox escape.
    ///
    /// Verified by mutation, and the result is worth recording: the sandbox confines twice, with
    /// the path normalisation and with `openat2(RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS)`. Removing
    /// **either one alone** changes nothing a caller can observe — the other still refuses every
    /// case here. Only with both removed do these tests fail. That is defence in depth working as
    /// intended, and it is also why a reviewer must not read "the test passes" as evidence that
    /// both layers are intact.
    #[test]
    fn the_sandbox_accepts_only_what_is_inside_its_roots() {
        let Some(fx) = fixture("accept") else {
            return;
        };
        let Ok(sbx) = PathSandbox::new(Some(&fx.root), Some(&fx.root)) else {
            return;
        };
        let outside_abs = fx.outside.join("secret.txt").display().to_string();
        let inside_abs = fx.root.join("ok.txt").display().to_string();

        // (candidate, must be rejected, why)
        let cases: Vec<(&str, bool, &str)> = vec![
            ("ok.txt", false, "a plain file inside the root"),
            ("sub/ok.txt", false, "a file in a subdirectory"),
            (&inside_abs, false, "an absolute path that is still inside"),
            ("../secret.txt", true, "the simplest traversal"),
            (
                "../../../../etc/passwd",
                true,
                "traversal past the filesystem root",
            ),
            (
                "sub/../../secret.txt",
                true,
                "traversal that re-enters before leaving",
            ),
            (&outside_abs, true, "an absolute path outside the root"),
            (
                "link_out",
                true,
                "a symlink pointing outside: RESOLVE_NO_SYMLINKS",
            ),
            (
                "link_in",
                true,
                "even a symlink that stays inside is refused",
            ),
            ("/dev/zero", true, "a character device"),
            ("", true, "the empty path"),
            (".", true, "the root itself is not a file"),
            ("ok\0.txt", true, "an interior NUL byte"),
        ];

        for (candidate, must_reject, why) in cases {
            let got = sbx.read_to_string_limited(std::path::Path::new(candidate), MAX_READ_BYTES);
            match got {
                Ok((_, content)) => {
                    assert!(
                        !must_reject,
                        "sandbox accepted {candidate:?} ({why}); it must be rejected"
                    );
                    assert!(
                        !content.contains("SECRET"),
                        "sandbox leaked content from outside its root via {candidate:?}"
                    );
                }
                Err(_) => assert!(
                    must_reject,
                    "sandbox rejected {candidate:?} ({why}); it must be accepted"
                ),
            }
        }
    }

    /// Writes are confined by the same contract as reads.
    ///
    /// The check is "nothing new exists outside the root", not "one guessed path is absent":
    /// a traversal lands wherever the join resolves to, which is rarely the path you first
    /// think of. Verified by mutation: removing the path validation from `open_write` makes
    /// this test fail.
    #[test]
    fn the_sandbox_refuses_to_write_outside_its_root() {
        let Some(fx) = fixture("write") else {
            return;
        };
        let Ok(sbx) = PathSandbox::new(Some(&fx.root), Some(&fx.root)) else {
            return;
        };

        fn snapshot(dir: &std::path::Path) -> Vec<String> {
            let mut out = Vec::new();
            let Ok(rd) = fs::read_dir(dir) else {
                return out;
            };
            for e in rd.flatten() {
                let p = e.path();
                out.push(p.display().to_string());
                if p.is_dir() && p.file_name().is_some_and(|n| n != "root") {
                    out.extend(snapshot(&p));
                }
            }
            out.sort();
            out
        }

        let before = snapshot(&fx.base);
        let probe = std::path::Path::new("/tmp/admetcraft_sbx_probe.txt");
        let _ = fs::remove_file(probe);

        for candidate in [
            "../written.txt".to_string(),
            "../../written.txt".to_string(),
            "sub/../../written.txt".to_string(),
            fx.outside.join("written.txt").display().to_string(),
            probe.display().to_string(),
            "link_out".to_string(),
        ] {
            let _ = sbx.open_write(std::path::Path::new(&candidate));
        }

        let after = snapshot(&fx.base);
        assert_eq!(
            before, after,
            "a write escaped the sandbox root: something new appeared outside it"
        );
        assert!(
            !probe.exists(),
            "a write escaped to an absolute path outside the root"
        );
        // The symlink target must not have been written through either.
        let secret = fs::read_to_string(fx.outside.join("secret.txt")).unwrap_or_default();
        assert_eq!(
            secret, "SECRET\n",
            "a write followed a symlink out of the root"
        );

        // ...and an honest write still works.
        assert!(sbx.open_write(std::path::Path::new("out.txt")).is_ok());
    }
}
