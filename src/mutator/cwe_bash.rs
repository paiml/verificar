//! CWE-targeted bash mutation for ShellSafetyBench
//!
//! Generates safe/unsafe shell script pairs with specific CWE patterns.
//! Used by bashrs ShellSafetyBench (spec S14.9, Steps 7.2/7.2b).
//!
//! ## Supported CWEs
//!
//! **In-distribution** (covered by bashrs linter):
//! - CWE-78: OS Command Injection (unquoted variables)
//! - CWE-94: Code Injection (eval usage)
//! - CWE-330: Insufficient Randomness ($RANDOM, $$)
//! - CWE-362: TOCTOU Race Condition (missing -p/-f flags)
//! - CWE-798: Hard-coded Credentials
//! - CWE-829: Inclusion of Untrusted Functionality (curl|bash)
//!
//! **Out-of-distribution** (eval-only, not in bashrs linter):
//! - CWE-426: Untrusted Search Path
//! - CWE-77: Command Injection (xargs without -0)
//! - CWE-116: Improper Output Encoding (log injection)
//! - CWE-250: Execution with Unnecessary Privileges

use serde::Serialize;

/// A CWE-targeted mutation: safe version + mutated unsafe version
#[derive(Debug, Clone, Serialize)]
pub struct CweMutation {
    /// Safe (original) script
    pub safe_script: String,
    /// Mutated (unsafe) script
    pub unsafe_script: String,
    /// CWE identifier (e.g., "CWE-78")
    pub cwe: String,
    /// CWE numeric ID
    pub cwe_id: u32,
    /// Human-readable vulnerability name
    pub vulnerability: String,
    /// Which mutation was applied
    pub mutation_description: String,
}

/// Generate CWE-targeted mutations for bash scripts.
///
/// Returns pairs of (safe, unsafe) scripts for each CWE pattern.
pub fn generate_cwe_mutations(cwe_targets: &[u32], count: usize, seed: u64) -> Vec<CweMutation> {
    let mut results = Vec::new();
    let per_cwe = count / cwe_targets.len().max(1);

    for &cwe_id in cwe_targets {
        let mutations = match cwe_id {
            78 => generate_cwe_78(per_cwe, seed),
            94 => generate_cwe_94(per_cwe, seed),
            330 => generate_cwe_330(per_cwe, seed),
            362 => generate_cwe_362(per_cwe, seed),
            798 => generate_cwe_798(per_cwe, seed),
            829 => generate_cwe_829(per_cwe, seed),
            377 => generate_cwe_377(per_cwe, seed),
            732 => generate_cwe_732(per_cwe, seed),
            // OOD CWEs
            426 => generate_cwe_426(per_cwe, seed),
            77 => generate_cwe_77(per_cwe, seed),
            116 => generate_cwe_116(per_cwe, seed),
            250 => generate_cwe_250(per_cwe, seed),
            _ => Vec::new(),
        };
        results.extend(mutations);
    }

    results.truncate(count);
    results
}

/// All in-distribution CWE IDs (covered by bashrs linter)
pub fn in_distribution_cwes() -> Vec<u32> {
    vec![78, 94, 330, 362, 377, 732, 798, 829]
}

/// All out-of-distribution CWE IDs (eval-only)
pub fn ood_cwes() -> Vec<u32> {
    vec![426, 77, 116, 250]
}

// --- In-Distribution CWEs ---

/// CWE-78: OS Command Injection via unquoted variables
fn generate_cwe_78(count: usize, _seed: u64) -> Vec<CweMutation> {
    let templates: Vec<(&str, &str, &str)> = vec![
        (
            "#!/bin/sh\nrm -rf \"${dir:?}\"/tmp",
            "#!/bin/sh\nrm -rf $dir/tmp",
            "Unquoted variable in rm command allows path injection",
        ),
        (
            "#!/bin/sh\ncp \"${src}\" \"${dst}\"",
            "#!/bin/sh\ncp $src $dst",
            "Unquoted variables in cp allow argument injection",
        ),
        (
            "#!/bin/sh\nfind \"${search_dir}\" -name '*.log'",
            "#!/bin/sh\nfind $search_dir -name '*.log'",
            "Unquoted variable in find allows path traversal",
        ),
        (
            "#!/bin/sh\ngrep \"${pattern}\" \"${file}\"",
            "#!/bin/sh\ngrep $pattern $file",
            "Unquoted grep arguments allow injection",
        ),
        (
            "#!/bin/sh\nchmod 644 \"${target_file}\"",
            "#!/bin/sh\nchmod 644 $target_file",
            "Unquoted variable in chmod allows arg injection",
        ),
        (
            "#!/bin/sh\nmv \"${old_name}\" \"${new_name}\"",
            "#!/bin/sh\nmv $old_name $new_name",
            "Unquoted variables in mv allow injection",
        ),
        (
            "#!/bin/sh\ntar czf backup.tar.gz \"${backup_dir}\"",
            "#!/bin/sh\ntar czf backup.tar.gz $backup_dir",
            "Unquoted variable in tar allows injection",
        ),
        (
            "#!/bin/sh\nssh \"${remote_host}\" 'uptime'",
            "#!/bin/sh\nssh $remote_host 'uptime'",
            "Unquoted variable in ssh allows host injection",
        ),
    ];

    templates
        .into_iter()
        .take(count)
        .map(|(safe, vuln, desc)| CweMutation {
            safe_script: safe.to_string(),
            unsafe_script: vuln.to_string(),
            cwe: "CWE-78".to_string(),
            cwe_id: 78,
            vulnerability: "OS Command Injection".to_string(),
            mutation_description: desc.to_string(),
        })
        .collect()
}

/// CWE-94: Code Injection via eval
fn generate_cwe_94(count: usize, _seed: u64) -> Vec<CweMutation> {
    let templates: Vec<(&str, &str, &str)> = vec![
        (
            "#!/bin/sh\necho \"$user_input\"",
            "#!/bin/sh\neval \"$user_input\"",
            "eval of user input allows arbitrary code execution",
        ),
        (
            "#!/bin/sh\nresult=$(echo \"$cmd_name\")",
            "#!/bin/sh\nresult=$(eval \"$cmd_name\")",
            "eval in command substitution allows injection",
        ),
        (
            "#!/bin/sh\necho \"Setting config: ${key}=${value}\"",
            "#!/bin/sh\neval \"${key}=${value}\"",
            "eval of config values allows code injection",
        ),
        (
            "#!/bin/sh\n. /etc/app/config.sh",
            "#!/bin/sh\n. \"$CONFIG_PATH\"",
            "source of variable path allows code injection",
        ),
    ];

    templates
        .into_iter()
        .take(count)
        .map(|(safe, vuln, desc)| CweMutation {
            safe_script: safe.to_string(),
            unsafe_script: vuln.to_string(),
            cwe: "CWE-94".to_string(),
            cwe_id: 94,
            vulnerability: "Code Injection".to_string(),
            mutation_description: desc.to_string(),
        })
        .collect()
}

/// CWE-330: Insufficient Randomness ($RANDOM, $$)
fn generate_cwe_330(count: usize, _seed: u64) -> Vec<CweMutation> {
    let templates: Vec<(&str, &str, &str)> = vec![
        (
            "#!/bin/sh\ntmpdir=$(mktemp -d)",
            "#!/bin/sh\ntmpdir=/tmp/work_$RANDOM",
            "$RANDOM is predictable (15-bit LFSR), use mktemp",
        ),
        (
            "#!/bin/sh\ntoken=$(head -c 32 /dev/urandom | base64)",
            "#!/bin/sh\ntoken=session_$$",
            "PID-based token is predictable",
        ),
        (
            "#!/bin/sh\nlog_file=$(mktemp /tmp/log.XXXXXX)",
            "#!/bin/sh\nlog_file=/tmp/log_$(date +%s)",
            "Timestamp-based filename is predictable",
        ),
        (
            "#!/bin/sh\nsalt=$(head -c 16 /dev/urandom | xxd -p)",
            "#!/bin/sh\nsalt=$RANDOM$RANDOM",
            "Double $RANDOM is only 30 bits of entropy",
        ),
    ];

    templates
        .into_iter()
        .take(count)
        .map(|(safe, vuln, desc)| CweMutation {
            safe_script: safe.to_string(),
            unsafe_script: vuln.to_string(),
            cwe: "CWE-330".to_string(),
            cwe_id: 330,
            vulnerability: "Insufficient Randomness".to_string(),
            mutation_description: desc.to_string(),
        })
        .collect()
}

/// CWE-362: TOCTOU Race Condition (missing -p/-f flags)
fn generate_cwe_362(count: usize, _seed: u64) -> Vec<CweMutation> {
    let templates: Vec<(&str, &str, &str)> = vec![
        (
            "#!/bin/sh\nmkdir -p \"${output_dir}\"",
            "#!/bin/sh\nmkdir \"${output_dir}\"",
            "mkdir without -p fails if dir exists (not idempotent)",
        ),
        (
            "#!/bin/sh\nrm -f \"${lockfile}\"",
            "#!/bin/sh\nrm \"${lockfile}\"",
            "rm without -f fails if file missing (not idempotent)",
        ),
        (
            "#!/bin/sh\nln -sf \"${target}\" \"${link_name}\"",
            "#!/bin/sh\nln -s \"${target}\" \"${link_name}\"",
            "ln -s without -f fails if link exists",
        ),
        (
            "#!/bin/sh\nmkdir -p /var/run/app && chmod 755 /var/run/app",
            "#!/bin/sh\nmkdir /var/run/app && chmod 755 /var/run/app",
            "mkdir without -p causes TOCTOU race on re-run",
        ),
    ];

    templates
        .into_iter()
        .take(count)
        .map(|(safe, vuln, desc)| CweMutation {
            safe_script: safe.to_string(),
            unsafe_script: vuln.to_string(),
            cwe: "CWE-362".to_string(),
            cwe_id: 362,
            vulnerability: "TOCTOU Race Condition".to_string(),
            mutation_description: desc.to_string(),
        })
        .collect()
}

/// CWE-798: Hard-coded Credentials
fn generate_cwe_798(count: usize, _seed: u64) -> Vec<CweMutation> {
    let templates: Vec<(&str, &str, &str)> = vec![
        (
            "#!/bin/sh\nDB_PASS=\"${DB_PASSWORD:?}\"",
            "#!/bin/sh\nDB_PASS=\"s3cret123\"",
            "Hard-coded database password",
        ),
        (
            "#!/bin/sh\nAPI_KEY=\"$(vault kv get -field=key secret/api)\"",
            "#!/bin/sh\nAPI_KEY=\"sk-abc123def456\"",
            "Hard-coded API key",
        ),
        (
            "#!/bin/sh\ntoken=\"${GITHUB_TOKEN:?}\"",
            "#!/bin/sh\ntoken=\"ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\"",
            "Hard-coded GitHub token",
        ),
    ];

    templates
        .into_iter()
        .take(count)
        .map(|(safe, vuln, desc)| CweMutation {
            safe_script: safe.to_string(),
            unsafe_script: vuln.to_string(),
            cwe: "CWE-798".to_string(),
            cwe_id: 798,
            vulnerability: "Hard-coded Credentials".to_string(),
            mutation_description: desc.to_string(),
        })
        .collect()
}

/// CWE-829: Inclusion of Untrusted Functionality (curl|bash)
fn generate_cwe_829(count: usize, _seed: u64) -> Vec<CweMutation> {
    let templates: Vec<(&str, &str, &str)> = vec![
        (
            "#!/bin/sh\ncurl -fsSL https://example.com/install.sh -o /tmp/install.sh\nsha256sum -c /tmp/install.sh.sha256\nsh /tmp/install.sh",
            "#!/bin/sh\ncurl -fsSL https://example.com/install.sh | sh",
            "Piping curl to shell bypasses integrity verification",
        ),
        (
            "#!/bin/sh\nwget -q https://example.com/setup.sh -O /tmp/setup.sh\nchmod +x /tmp/setup.sh\n/tmp/setup.sh",
            "#!/bin/sh\nwget -q -O- https://example.com/setup.sh | bash",
            "Piping wget to bash allows MitM code execution",
        ),
    ];

    templates
        .into_iter()
        .take(count)
        .map(|(safe, vuln, desc)| CweMutation {
            safe_script: safe.to_string(),
            unsafe_script: vuln.to_string(),
            cwe: "CWE-829".to_string(),
            cwe_id: 829,
            vulnerability: "Inclusion of Untrusted Functionality".to_string(),
            mutation_description: desc.to_string(),
        })
        .collect()
}

/// CWE-377: Insecure Temporary File
fn generate_cwe_377(count: usize, _seed: u64) -> Vec<CweMutation> {
    let templates: Vec<(&str, &str, &str)> = vec![
        (
            "#!/bin/sh\ntmp=$(mktemp)",
            "#!/bin/sh\ntmp=/tmp/myapp_tmp",
            "Predictable temp file allows symlink attack",
        ),
        (
            "#!/bin/sh\nworkdir=$(mktemp -d)",
            "#!/bin/sh\nworkdir=/tmp/workdir",
            "Predictable temp directory allows pre-creation attack",
        ),
    ];

    templates
        .into_iter()
        .take(count)
        .map(|(safe, vuln, desc)| CweMutation {
            safe_script: safe.to_string(),
            unsafe_script: vuln.to_string(),
            cwe: "CWE-377".to_string(),
            cwe_id: 377,
            vulnerability: "Insecure Temporary File".to_string(),
            mutation_description: desc.to_string(),
        })
        .collect()
}

/// CWE-732: Incorrect Permission Assignment
fn generate_cwe_732(count: usize, _seed: u64) -> Vec<CweMutation> {
    let templates: Vec<(&str, &str, &str)> = vec![
        (
            "#!/bin/sh\nchmod 600 /etc/app/secrets.conf",
            "#!/bin/sh\nchmod 777 /etc/app/secrets.conf",
            "World-writable permissions on secrets file",
        ),
        (
            "#!/bin/sh\nchmod 700 /var/run/app",
            "#!/bin/sh\nchmod 666 /var/run/app",
            "World-readable/writable runtime directory",
        ),
    ];

    templates
        .into_iter()
        .take(count)
        .map(|(safe, vuln, desc)| CweMutation {
            safe_script: safe.to_string(),
            unsafe_script: vuln.to_string(),
            cwe: "CWE-732".to_string(),
            cwe_id: 732,
            vulnerability: "Incorrect Permission Assignment".to_string(),
            mutation_description: desc.to_string(),
        })
        .collect()
}

// --- Out-of-Distribution CWEs (eval-only) ---

/// CWE-426: Untrusted Search Path
fn generate_cwe_426(count: usize, _seed: u64) -> Vec<CweMutation> {
    let templates: Vec<(&str, &str, &str)> = vec![
        (
            "#!/bin/sh\n/usr/bin/python3 script.py",
            "#!/bin/sh\npython3 script.py",
            "Relative command path allows PATH hijacking",
        ),
        (
            "#!/bin/sh\nPATH=/usr/bin:/bin\nexport PATH\ngcc -o app app.c",
            "#!/bin/sh\nPATH=.:$PATH\nexport PATH\ngcc -o app app.c",
            "Current directory in PATH allows trojan binary execution",
        ),
        (
            "#!/bin/sh\n/bin/ls /var/log",
            "#!/bin/sh\nls /var/log",
            "Relative 'ls' can be hijacked via PATH manipulation",
        ),
        (
            "#!/bin/sh\nLD_LIBRARY_PATH=/usr/lib\nexport LD_LIBRARY_PATH\n./app",
            "#!/bin/sh\nLD_LIBRARY_PATH=.:/usr/lib\nexport LD_LIBRARY_PATH\n./app",
            "Current dir in LD_LIBRARY_PATH allows library hijack",
        ),
    ];

    templates
        .into_iter()
        .take(count)
        .map(|(safe, vuln, desc)| CweMutation {
            safe_script: safe.to_string(),
            unsafe_script: vuln.to_string(),
            cwe: "CWE-426".to_string(),
            cwe_id: 426,
            vulnerability: "Untrusted Search Path".to_string(),
            mutation_description: desc.to_string(),
        })
        .collect()
}

/// CWE-77: Command Injection (xargs, indirect execution)
fn generate_cwe_77(count: usize, _seed: u64) -> Vec<CweMutation> {
    let templates: Vec<(&str, &str, &str)> = vec![
        (
            "#!/bin/sh\nfind . -name '*.txt' -print0 | xargs -0 rm",
            "#!/bin/sh\nfind . -name '*.txt' | xargs rm",
            "xargs without -0 allows filename injection via newlines/spaces",
        ),
        (
            "#!/bin/sh\nfind /tmp -type f -print0 | xargs -0 chmod 644",
            "#!/bin/sh\nfind /tmp -type f | xargs chmod 644",
            "Missing -print0/-0 allows injection via crafted filenames",
        ),
        (
            "#!/bin/sh\nprintf '%s\\0' \"$@\" | xargs -0 grep pattern",
            "#!/bin/sh\necho \"$@\" | xargs grep pattern",
            "Word-splitting user args through xargs without null delimiter",
        ),
        (
            "#!/bin/sh\nwhile IFS= read -r file; do\n  process \"$file\"\ndone < filelist.txt",
            "#!/bin/sh\nfor file in $(cat filelist.txt); do\n  process $file\ndone",
            "Command substitution + unquoted var allows injection via filenames",
        ),
    ];

    templates
        .into_iter()
        .take(count)
        .map(|(safe, vuln, desc)| CweMutation {
            safe_script: safe.to_string(),
            unsafe_script: vuln.to_string(),
            cwe: "CWE-77".to_string(),
            cwe_id: 77,
            vulnerability: "Command Injection".to_string(),
            mutation_description: desc.to_string(),
        })
        .collect()
}

/// CWE-116: Improper Output Encoding (log injection)
fn generate_cwe_116(count: usize, _seed: u64) -> Vec<CweMutation> {
    let templates: Vec<(&str, &str, &str)> = vec![
        (
            "#!/bin/sh\nprintf '%s\\n' \"$user_input\" >> /var/log/app.log",
            "#!/bin/sh\necho $user_input >> /var/log/app.log",
            "Unquoted echo allows log injection via newlines and escape sequences",
        ),
        (
            "#!/bin/sh\nprintf 'User: %s Action: %s\\n' \"$user\" \"$action\" >> audit.log",
            "#!/bin/sh\necho \"User: $user Action: $action\" >> audit.log",
            "Unvalidated user/action fields allow log forging",
        ),
        (
            "#!/bin/sh\nprintf '%s\\n' \"${msg}\" | tee -a output.log",
            "#!/bin/sh\necho -e \"$msg\" | tee -a output.log",
            "echo -e interprets escape sequences from untrusted input",
        ),
    ];

    templates
        .into_iter()
        .take(count)
        .map(|(safe, vuln, desc)| CweMutation {
            safe_script: safe.to_string(),
            unsafe_script: vuln.to_string(),
            cwe: "CWE-116".to_string(),
            cwe_id: 116,
            vulnerability: "Improper Output Encoding".to_string(),
            mutation_description: desc.to_string(),
        })
        .collect()
}

/// CWE-250: Execution with Unnecessary Privileges
fn generate_cwe_250(count: usize, _seed: u64) -> Vec<CweMutation> {
    let templates: Vec<(&str, &str, &str)> = vec![
        (
            "#!/bin/sh\ninstall -m 644 config.conf /etc/app/",
            "#!/bin/sh\nsudo cp config.conf /etc/app/",
            "Unnecessary sudo for file copy when install suffices",
        ),
        (
            "#!/bin/sh\nsu -c 'systemctl restart app' appuser",
            "#!/bin/sh\nsudo systemctl restart app",
            "Running as root when specific user suffices",
        ),
        (
            "#!/bin/sh\nchown appuser:appgroup /var/run/app.pid",
            "#!/bin/sh\nsudo chmod 777 /var/run/app.pid",
            "Using sudo + world-writable instead of proper ownership",
        ),
        (
            "#!/bin/sh\ncap_add NET_BIND_SERVICE /usr/bin/app",
            "#!/bin/sh\nsudo /usr/bin/app",
            "Running entire app as root instead of adding specific capability",
        ),
    ];

    templates
        .into_iter()
        .take(count)
        .map(|(safe, vuln, desc)| CweMutation {
            safe_script: safe.to_string(),
            unsafe_script: vuln.to_string(),
            cwe: "CWE-250".to_string(),
            cwe_id: 250,
            vulnerability: "Execution with Unnecessary Privileges".to_string(),
            mutation_description: desc.to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_cwe_78_mutations() {
        let mutations = generate_cwe_78(5, 42);
        assert!(!mutations.is_empty());
        assert!(mutations.len() <= 5);
        for m in &mutations {
            assert_eq!(m.cwe, "CWE-78");
            assert_eq!(m.cwe_id, 78);
            assert!(!m.safe_script.is_empty());
            assert!(!m.unsafe_script.is_empty());
            assert_ne!(m.safe_script, m.unsafe_script);
        }
    }

    #[test]
    fn test_generate_cwe_94_mutations() {
        let mutations = generate_cwe_94(3, 42);
        assert!(!mutations.is_empty());
        for m in &mutations {
            assert_eq!(m.cwe, "CWE-94");
            assert!(m.unsafe_script.contains("eval") || m.unsafe_script.contains(". \"$"));
        }
    }

    #[test]
    fn test_generate_ood_cwe_426() {
        let mutations = generate_cwe_426(4, 42);
        assert!(!mutations.is_empty());
        for m in &mutations {
            assert_eq!(m.cwe, "CWE-426");
            assert_eq!(m.cwe_id, 426);
        }
    }

    #[test]
    fn test_generate_ood_cwe_77() {
        let mutations = generate_cwe_77(4, 42);
        assert!(!mutations.is_empty());
        for m in &mutations {
            assert_eq!(m.cwe, "CWE-77");
        }
    }

    #[test]
    fn test_generate_ood_cwe_116() {
        let mutations = generate_cwe_116(3, 42);
        assert!(!mutations.is_empty());
        for m in &mutations {
            assert_eq!(m.cwe, "CWE-116");
        }
    }

    #[test]
    fn test_generate_ood_cwe_250() {
        let mutations = generate_cwe_250(4, 42);
        assert!(!mutations.is_empty());
        for m in &mutations {
            assert_eq!(m.cwe, "CWE-250");
        }
    }

    #[test]
    fn test_generate_cwe_mutations_mixed() {
        let mutations = generate_cwe_mutations(&[78, 94, 426, 77], 20, 42);
        assert!(!mutations.is_empty());
        assert!(mutations.len() <= 20);
        // Should have multiple CWE types
        let cwes: std::collections::HashSet<&str> =
            mutations.iter().map(|m| m.cwe.as_str()).collect();
        assert!(cwes.len() > 1);
    }

    #[test]
    fn test_in_distribution_cwes() {
        let ids = in_distribution_cwes();
        assert!(ids.contains(&78));
        assert!(ids.contains(&94));
        assert!(ids.contains(&330));
        assert!(ids.contains(&362));
    }

    #[test]
    fn test_ood_cwes() {
        let ids = ood_cwes();
        assert!(ids.contains(&426));
        assert!(ids.contains(&77));
        assert!(ids.contains(&116));
        assert!(ids.contains(&250));
    }

    #[test]
    fn test_ood_disjoint_from_in_distribution() {
        let id = in_distribution_cwes();
        let ood = ood_cwes();
        for cwe in &ood {
            assert!(
                !id.contains(cwe),
                "OOD CWE {cwe} should not be in-distribution"
            );
        }
    }

    #[test]
    fn test_cwe_mutation_serialization() {
        let mutations = generate_cwe_78(1, 42);
        let json = serde_json::to_string(&mutations[0]).expect("serialize");
        assert!(json.contains("CWE-78"));
        assert!(json.contains("safe_script"));
        assert!(json.contains("unsafe_script"));
    }
}
