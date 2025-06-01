"""
Large Scale Dataset Generator for Login Event Classification
Generates 20,000 diverse, realistic log entries
"""

import pandas as pd
import random
from datetime import datetime, timedelta
import ipaddress
import hashlib
from typing import List, Dict, Tuple

class LogGenerator:
    """Generate realistic log entries at scale"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        
        # Common usernames
        self.usernames = [
            'admin', 'root', 'administrator', 'user', 'test', 'guest', 'demo',
            'john', 'alice', 'bob', 'mary', 'david', 'sarah', 'michael', 'jennifer',
            'postgres', 'mysql', 'oracle', 'dbadmin', 'dbuser', 'app_user',
            'www-data', 'nginx', 'apache', 'web_admin', 'deploy', 'jenkins',
            'git', 'gitlab', 'backup', 'monitoring', 'nagios', 'zabbix',
            'service', 'system', 'operator', 'support', 'helpdesk', 'analyst',
            'developer', 'devops', 'sysadmin', 'netadmin', 'security', 'audit'
        ]
        
        # Common service/system names
        self.services = [
            'sshd', 'systemd', 'systemd-logind', 'login', 'su', 'sudo',
            'apache2', 'nginx', 'httpd', 'tomcat', 'java', 'node',
            'postgres', 'mysql', 'mariadb', 'mongodb', 'redis', 'oracle',
            'vsftpd', 'proftpd', 'ftpd', 'sftp-server',
            'openvpn', 'ipsec', 'pptp', 'l2tp',
            'docker', 'containerd', 'kubelet', 'kube-apiserver',
            'postfix', 'dovecot', 'sendmail', 'exim'
        ]
        
        # Hostnames
        self.hostnames = [
            'web-prod-01', 'web-prod-02', 'app-server-01', 'db-master',
            'db-slave-01', 'mail-server', 'file-server', 'backup-server',
            'monitoring-01', 'jenkins-master', 'gitlab-runner-01',
            'k8s-master-01', 'k8s-worker-01', 'k8s-worker-02',
            'vpn-gateway', 'proxy-01', 'proxy-02', 'loadbalancer',
            'auth-server', 'ldap-master', 'ad-controller-01'
        ]
        
        # Domains
        self.domains = [
            'CORPORATE', 'INTERNAL', 'PROD', 'DEV', 'TEST',
            'company.com', 'internal.local', 'corp.local'
        ]
        
        # Ports
        self.ssh_ports = [22, 2222, 22022, 2200, 8022]
        self.common_ports = [80, 443, 3306, 5432, 3389, 21, 25, 110, 143]
        
        # IP ranges
        self.ip_ranges = [
            '192.168.', '10.0.', '172.16.', '172.17.', '172.18.',
            '203.0.113.', '198.51.100.', '100.64.'
        ]
        
        # Failure reasons
        self.failure_reasons = [
            'Invalid password', 'Bad password', 'Authentication failure',
            'Account locked', 'Account disabled', 'Account expired',
            'Invalid credentials', 'Permission denied', 'Access denied',
            'Maximum attempts exceeded', 'User not found', 'Invalid user'
        ]
        
        # Success methods
        self.auth_methods = [
            'password', 'publickey', 'keyboard-interactive', 'gssapi-with-mic',
            'certificate', 'token', 'kerberos', 'ldap', 'radius'
        ]

    def generate_ip(self) -> str:
        """Generate a random IP address"""
        if random.random() < 0.1:  # 10% chance for IPv6
            return f"2001:db8::{random.randint(1, 9999):x}"
        else:
            prefix = random.choice(self.ip_ranges)
            return f"{prefix}{random.randint(1, 254)}.{random.randint(1, 254)}"
    
    def generate_timestamp(self) -> str:
        """Generate a random timestamp in various formats"""
        # Random time within last 30 days
        base_time = datetime.now() - timedelta(days=random.randint(0, 30))
        timestamp = base_time - timedelta(
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Different timestamp formats
        formats = [
            "%b %d %H:%M:%S",  # Jan 01 12:34:56
            "%Y-%m-%dT%H:%M:%S.%fZ",  # 2024-01-01T12:34:56.789Z
            "%Y/%m/%d %H:%M:%S",  # 2024/01/01 12:34:56
            "[%d/%b/%Y:%H:%M:%S +0000]",  # [01/Jan/2024:12:34:56 +0000]
            "%Y-%m-%d %H:%M:%S",  # 2024-01-01 12:34:56
        ]
        
        return timestamp.strftime(random.choice(formats))
    
    def generate_pid(self) -> int:
        """Generate a random process ID"""
        return random.randint(1000, 65535)
    
    def generate_successful_ssh_log(self) -> str:
        """Generate successful SSH login log"""
        templates = [
            "sshd[{pid}]: Accepted {method} for {user} from {ip} port {port} ssh2",
            "sshd[{pid}]: Accepted {method} for {user} from {ip} port {port} ssh2: RSA SHA256:{hash}",
            "sshd[{pid}]: pam_unix(sshd:session): session opened for user {user} by (uid=0)",
            "sshd[{pid}]: User {user} from {ip} authenticated successfully",
            "sshd[{pid}]: Successful authentication for {user} from {ip} port {port}",
        ]
        
        return random.choice(templates).format(
            pid=self.generate_pid(),
            method=random.choice(self.auth_methods[:3]),
            user=random.choice(self.usernames),
            ip=self.generate_ip(),
            port=random.randint(40000, 65535),
            hash=hashlib.md5(str(random.random()).encode()).hexdigest()[:12]
        )
    
    def generate_failed_ssh_log(self) -> str:
        """Generate failed SSH login log"""
        templates = [
            "sshd[{pid}]: Failed password for {user} from {ip} port {port} ssh2",
            "sshd[{pid}]: Failed password for invalid user {user} from {ip} port {port} ssh2",
            "sshd[{pid}]: Invalid user {user} from {ip} port {port}",
            "sshd[{pid}]: Connection closed by authenticating user {user} {ip} port {port} [preauth]",
            "sshd[{pid}]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost={ip} user={user}",
            "sshd[{pid}]: error: maximum authentication attempts exceeded for {user} from {ip} port {port} ssh2 [preauth]",
            "sshd[{pid}]: Disconnecting invalid user {user} {ip} port {port}: Too many authentication failures",
        ]
        
        return random.choice(templates).format(
            pid=self.generate_pid(),
            user=random.choice(self.usernames),
            ip=self.generate_ip(),
            port=random.randint(40000, 65535)
        )
    
    def generate_successful_system_log(self) -> str:
        """Generate successful system authentication log"""
        templates = [
            "systemd-logind[{pid}]: New session {session} of user {user}.",
            "su[{pid}]: Successful su for {user} by {from_user}",
            "su[{pid}]: + pts/{pts} {from_user}:{user}",
            "sudo: {from_user} : TTY=pts/{pts} ; PWD=/home/{from_user} ; USER={user} ; COMMAND={cmd}",
            "sudo: pam_unix(sudo:session): session opened for user {user} by {from_user}(uid={uid})",
            "gdm-password][{pid}]: pam_unix(gdm-password:session): session opened for user {user} by (uid=0)",
            "login[{pid}]: pam_unix(login:session): session opened for user {user} by LOGIN(uid=0)",
        ]
        
        return random.choice(templates).format(
            pid=self.generate_pid(),
            session=random.randint(1, 999),
            user=random.choice(self.usernames),
            from_user=random.choice(self.usernames),
            pts=random.randint(0, 9),
            cmd=random.choice(['/bin/bash', '/bin/su', '/usr/bin/vim', '/bin/cat /etc/passwd']),
            uid=random.randint(1000, 2000)
        )
    
    def generate_failed_system_log(self) -> str:
        """Generate failed system authentication log"""
        templates = [
            "su[{pid}]: FAILED su for {user} by {from_user}",
            "su[{pid}]: - pts/{pts} {from_user}:{user}",
            "sudo: {from_user} : {attempts} incorrect password attempts ; TTY=pts/{pts} ; PWD=/home/{from_user} ; USER={user} ; COMMAND={cmd}",
            "login[{pid}]: FAILED LOGIN ({attempts}) on '/dev/tty{tty}' FOR '{user}', Authentication failure",
            "pam_unix(login:auth): authentication failure; logname= uid=0 euid=0 tty=tty{tty} ruser= rhost=  user={user}",
        ]
        
        return random.choice(templates).format(
            pid=self.generate_pid(),
            user=random.choice(self.usernames),
            from_user=random.choice(self.usernames),
            pts=random.randint(0, 9),
            tty=random.randint(1, 6),
            attempts=random.randint(1, 5),
            cmd=random.choice(['/bin/bash', '/bin/su', '/usr/bin/passwd'])
        )
    
    def generate_successful_web_log(self) -> str:
        """Generate successful web authentication log"""
        templates = [
            "apache2[{pid}]: Auth: user {user}@{domain} : authentication successful",
            "nginx[{pid}]: User '{user}' logged in successfully from {ip}",
            "[INFO] SecurityManager - Login successful for user: {user}, IP: {ip}",
            "tomcat[{pid}]: User [{user}] successfully authenticated via {method}",
            "{service}[{pid}]: Successful login: user={user} ip={ip} session={session}",
        ]
        
        return random.choice(templates).format(
            pid=self.generate_pid(),
            user=random.choice(self.usernames),
            domain=random.choice(self.domains),
            ip=self.generate_ip(),
            method=random.choice(['LDAP', 'OAuth', 'SAML', 'local']),
            service=random.choice(['httpd', 'apache2', 'nginx', 'tomcat']),
            session=hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
        )
    
    def generate_failed_web_log(self) -> str:
        """Generate failed web authentication log"""
        templates = [
            "apache2[{pid}]: Auth: user {user}@{domain} : authentication failure for '{path}': {reason}",
            "nginx[{pid}]: Login failed for user '{user}' from {ip} - {reason}",
            "[ERROR] SecurityManager - Login failed for user: {user}, IP: {ip}, Reason: {reason}",
            "tomcat[{pid}]: Authentication failed for user [{user}] - {reason}",
            "{service}[{pid}]: Failed login attempt: user={user} ip={ip} reason='{reason}'",
        ]
        
        return random.choice(templates).format(
            pid=self.generate_pid(),
            user=random.choice(self.usernames),
            domain=random.choice(self.domains),
            ip=self.generate_ip(),
            path=random.choice(['/admin', '/login', '/api/auth', '/dashboard']),
            reason=random.choice(self.failure_reasons),
            service=random.choice(['httpd', 'apache2', 'nginx', 'tomcat'])
        )
    
    def generate_successful_database_log(self) -> str:
        """Generate successful database authentication log"""
        templates = [
            "postgres[{pid}]: LOG:  connection authorized: user={user} database={db}",
            "mysql[{pid}]: Access granted for user '{user}'@'{ip}' (using password: YES)",
            "Oracle Audit: LOGON: USER: '{user}' PRIVILEGE: '{priv}' SUCCESS",
            "mongodb[{pid}]: Successfully authenticated as principal {user} on {db}",
            "{service}[{pid}]: User {user} connected to database {db} from {ip}",
        ]
        
        return random.choice(templates).format(
            pid=self.generate_pid(),
            user=random.choice(['dbadmin', 'app_user', 'readonly', 'backup', 'replication']),
            db=random.choice(['production', 'development', 'test', 'admin', 'master']),
            ip=self.generate_ip(),
            priv=random.choice(['SYSDBA', 'NORMAL', 'SYSOPER']),
            service=random.choice(['postgres', 'mysql', 'mariadb'])
        )
    
    def generate_failed_database_log(self) -> str:
        """Generate failed database authentication log"""
        templates = [
            "postgres[{pid}]: FATAL:  password authentication failed for user '{user}'",
            "mysql[{pid}]: Access denied for user '{user}'@'{ip}' (using password: {pwd})",
            "Oracle Audit: LOGON: USER: '{user}' PRIVILEGE: '{priv}' FAILURE: ORA-01017: invalid username/password",
            "mongodb[{pid}]: Failed to authenticate as principal {user} on {db} - AuthenticationFailed",
            "{service}[{pid}]: Authentication failed for user {user} from {ip}: {reason}",
        ]
        
        return random.choice(templates).format(
            pid=self.generate_pid(),
            user=random.choice(['dbadmin', 'root', 'sa', 'admin', 'test']),
            db=random.choice(['production', 'admin', 'master']),
            ip=self.generate_ip(),
            pwd=random.choice(['YES', 'NO']),
            priv=random.choice(['SYSDBA', 'NORMAL']),
            service=random.choice(['postgres', 'mysql', 'mariadb']),
            reason=random.choice(['Invalid password', 'User not found', 'Account locked'])
        )
    
    def generate_successful_windows_log(self) -> str:
        """Generate successful Windows authentication log"""
        templates = [
            "Logon Type: {logon_type}. New Logon: Security ID: {sid} Account Name: {user} Account Domain: {domain}",
            "Special privileges assigned to new logon: User: {domain}\\{user} Privileges: {privs}",
            "An account was successfully logged on. Subject: Security ID: {sid} Logon Type: {logon_type}",
            "Windows Logon: User {domain}\\{user} logged on successfully from {ip}",
        ]
        
        return random.choice(templates).format(
            logon_type=random.choice([2, 3, 7, 10, 11]),  # Interactive, Network, Unlock, RemoteInteractive, CachedInteractive
            sid=f"S-1-5-21-{random.randint(100000, 999999)}-{random.randint(1000, 9999)}",
            user=random.choice(self.usernames),
            domain=random.choice(['CORPORATE', 'INTERNAL', 'WORKGROUP']),
            privs=random.choice(['SeBackupPrivilege', 'SeRestorePrivilege', 'SeDebugPrivilege']),
            ip=self.generate_ip()
        )
    
    def generate_failed_windows_log(self) -> str:
        """Generate failed Windows authentication log"""
        templates = [
            "Logon Failure: Reason: {reason}. User Name: {user} Domain: {domain}",
            "An account failed to log on. Subject: Security ID: {sid} Failure Reason: {reason}",
            "Audit Failure: Logon Type: {logon_type} Failure Information: Failure Reason: {reason}",
            "Windows Logon Failed: {domain}\\{user} from {ip} - {reason}",
        ]
        
        return random.choice(templates).format(
            reason=random.choice(self.failure_reasons),
            user=random.choice(self.usernames),
            domain=random.choice(['CORPORATE', 'INTERNAL', 'WORKGROUP']),
            sid=f"S-1-5-21-{random.randint(100000, 999999)}-{random.randint(1000, 9999)}",
            logon_type=random.choice([2, 3, 7, 10]),
            ip=self.generate_ip()
        )
    
    def generate_non_login_log(self) -> str:
        """Generate non-login related log"""
        templates = [
            # System events
            "kernel: Out of memory: Kill process {pid} ({process}) score {score} or sacrifice child",
            "kernel: CPU{cpu}: Temperature above threshold, cpu clock throttled",
            "kernel: [{timestamp}] usb {bus}-{device}: new high-speed USB device number {num} using xhci_hcd",
            "systemd[1]: Started {service} - {description}.",
            "systemd[1]: Stopped {service} - {description}.",
            
            # Cron jobs
            "CRON[{pid}]: ({user}) CMD ({command})",
            "anacron[{pid}]: Job '{job}' started",
            
            # Network events
            "dhclient[{pid}]: DHCPREQUEST on {iface} to {ip} port 67",
            "NetworkManager[{pid}]: <info> ({iface}): device state change: {state1} -> {state2}",
            "kernel: [UFW BLOCK] IN={iface} OUT= MAC={mac} SRC={src} DST={dst}",
            
            # Service logs
            "apache2[{pid}]: {ip} - - [{timestamp}] '{method} {path} HTTP/1.1' {status} {bytes}",
            "nginx[{pid}]: {ip} - - [{timestamp}] '{method} {path} HTTP/1.1' {status} {bytes}",
            
            # Database operations
            "postgres[{pid}]: LOG: checkpoint starting: {reason}",
            "mysql[{pid}]: InnoDB: {operation}",
            
            # Docker/Container
            "docker[{pid}]: Container {container_id} {action}",
            "containerd[{pid}]: time='{timestamp}' level=info msg='{message}'",
            
            # Email
            "postfix/smtp[{pid}]: {queue_id}: to=<{email}>, relay={relay}, status={status}",
            "dovecot[{pid}]: imap({user}): Disconnected: {reason} in={inbytes} out={outbytes}",
        ]
        
        # Generate based on template
        template = random.choice(templates)
        
        # Helper values
        log_params = {
            'pid': self.generate_pid(),
            'process': random.choice(['java', 'python', 'node', 'chrome', 'firefox']),
            'score': random.randint(100, 999),
            'cpu': random.randint(0, 7),
            'timestamp': random.randint(100000, 999999),
            'bus': random.randint(1, 4),
            'device': random.randint(1, 8),
            'num': random.randint(1, 10),
            'service': random.choice(['Docker', 'MySQL', 'Apache', 'Nginx', 'PostgreSQL']),
            'description': random.choice(['Web Server', 'Database Server', 'Application Container', 'Message Queue']),
            'user': random.choice(self.usernames + ['root', 'www-data']),
            'command': random.choice(['/usr/local/bin/backup.sh', 'php /var/www/cron.php', '/usr/bin/updatedb']),
            'job': random.choice(['cron.daily', 'cron.weekly', 'cron.monthly']),
            'iface': random.choice(['eth0', 'eth1', 'ens33', 'wlan0']),
            'ip': self.generate_ip(),
            'state1': 'ip-config',
            'state2': 'activated',
            'mac': ':'.join([f'{random.randint(0, 255):02x}' for _ in range(6)]),
            'src': self.generate_ip(),
            'dst': self.generate_ip(),
            'method': random.choice(['GET', 'POST', 'PUT', 'DELETE']),
            'path': random.choice(['/index.html', '/api/data', '/images/logo.png', '/css/style.css']),
            'status': random.choice([200, 201, 204, 301, 302, 400, 401, 403, 404, 500]),
            'bytes': random.randint(100, 100000),
            'reason': random.choice(['time', 'xlog', 'checkpoint', 'shutdown']),
            'operation': random.choice(['Starting crash recovery', 'Buffer pool initialized', 'Log file created']),
            'container_id': hashlib.md5(str(random.random()).encode()).hexdigest()[:12],
            'action': random.choice(['started', 'stopped', 'created', 'removed']),
            'message': random.choice(['Container exec started', 'Container stopped', 'Image pulled']),
            'queue_id': hashlib.md5(str(random.random()).encode()).hexdigest()[:10].upper(),
            'email': f"{random.choice(self.usernames)}@example.com",
            'relay': f"mail.{random.choice(['gmail', 'yahoo', 'outlook'])}.com",
            'status': random.choice(['sent', 'bounced', 'deferred']),
            'inbytes': random.randint(100, 10000),
            'outbytes': random.randint(100, 50000),
        }
        
        return template.format(**log_params)
    
    def add_timestamp_and_hostname(self, log: str) -> str:
        """Add timestamp and hostname to log entry"""
        if random.random() < 0.7:  # 70% chance to add timestamp and hostname
            timestamp = self.generate_timestamp()
            hostname = random.choice(self.hostnames)
            return f"{timestamp} {hostname} {log}"
        return log
    
    def generate_dataset(self, total_logs: int = 20000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset with specified number of logs"""
        
        # Calculate distribution (roughly equal)
        success_count = total_logs // 3
        failed_count = total_logs // 3
        non_login_count = total_logs - success_count - failed_count
        
        print(f"Generating {total_logs} logs...")
        print(f"- Successful logins: {success_count}")
        print(f"- Failed logins: {failed_count}")
        print(f"- Non-login events: {non_login_count}")
        
        # Generate successful login logs
        success_logs = []
        generators = [
            self.generate_successful_ssh_log,
            self.generate_successful_system_log,
            self.generate_successful_web_log,
            self.generate_successful_database_log,
            self.generate_successful_windows_log
        ]
        
        for i in range(success_count):
            if i % 1000 == 0:
                print(f"Generated {i}/{success_count} successful login logs...")
            
            generator = random.choice(generators)
            log = generator()
            log = self.add_timestamp_and_hostname(log)
            success_logs.append(log)
        
        # Generate failed login logs
        failed_logs = []
        generators = [
            self.generate_failed_ssh_log,
            self.generate_failed_system_log,
            self.generate_failed_web_log,
            self.generate_failed_database_log,
            self.generate_failed_windows_log
        ]
        
        for i in range(failed_count):
            if i % 1000 == 0:
                print(f"Generated {i}/{failed_count} failed login logs...")
            
            generator = random.choice(generators)
            log = generator()
            log = self.add_timestamp_and_hostname(log)
            failed_logs.append(log)
        
        # Generate non-login logs
        non_login_logs = []
        for i in range(non_login_count):
            if i % 1000 == 0:
                print(f"Generated {i}/{non_login_count} non-login logs...")
            
            log = self.generate_non_login_log()
            log = self.add_timestamp_and_hostname(log)
            non_login_logs.append(log)
        
        # Create DataFrames
        def detect_asset_type(log_text):
            """Automatically detect asset type from log content"""
            log_lower = log_text.lower()
            
            # Define patterns for asset detection
            if 'sshd[' in log_lower or 'ssh2' in log_lower or 'openssh' in log_lower:
                return 'SSH'
            elif 'systemd' in log_lower or 'pam_unix' in log_lower or 'su[' in log_lower or 'sudo:' in log_lower:
                return 'Linux'
            elif 'logon type:' in log_lower or 'security id:' in log_lower or 'windows' in log_lower:
                return 'Windows'
            elif 'postgres[' in log_lower or 'mysql[' in log_lower or 'oracle' in log_lower or 'mongodb' in log_lower:
                return 'Database'
            elif 'apache2[' in log_lower or 'nginx[' in log_lower or 'iis' in log_lower:
                return 'WebServer'
            elif 'ftpd[' in log_lower or 'vsftpd[' in log_lower or 'proftpd[' in log_lower:
                return 'FTP'
            elif 'openvpn[' in log_lower or 'vpn-server[' in log_lower:
                return 'VPN'
            elif 'docker[' in log_lower or 'containerd[' in log_lower or 'kube-apiserver[' in log_lower:
                return 'Container'
            elif 'cron[' in log_lower or 'crond[' in log_lower:
                return 'Cron'
            elif 'kernel:' in log_lower:
                return 'System'
            elif 'postfix' in log_lower or 'dovecot' in log_lower or 'sendmail' in log_lower:
                return 'Email'
            elif 'ldap[' in log_lower or 'winbind[' in log_lower:
                return 'LDAP'
            elif any(auth in log_lower for auth in ['auth:', 'authentication', 'login', 'logon']):
                return 'Application'
            else:
                return 'Other'
        
        df_success = pd.DataFrame({
            'ID': range(1, len(success_logs) + 1),
            'Asset': [detect_asset_type(log) for log in success_logs],
            'Log': success_logs
        })
        
        df_failed = pd.DataFrame({
            'ID': range(1, len(failed_logs) + 1),
            'Asset': [detect_asset_type(log) for log in failed_logs],
            'Log': failed_logs
        })
        
        df_non_login = pd.DataFrame({
            'ID': range(1, len(non_login_logs) + 1),
            'Asset': [detect_asset_type(log) for log in non_login_logs],
            'Log': non_login_logs
        })
        
        print("\nDataset generation complete!")
        
        return df_success, df_failed, df_non_login

def main():
    """Generate and save large dataset"""
    
    # Initialize generator
    generator = LogGenerator(seed=42)  # Use seed for reproducibility
    
    # Generate 20,000 logs
    df_success, df_failed, df_non_login = generator.generate_dataset(total_logs=40000)
    
    # Save to CSV files
    print("\nSaving to CSV files...")
    df_success.to_csv('data/created-logs/successful_login_logs_20k.csv', index=False)
    df_failed.to_csv('data/created-logs/failed_login_logs_20k.csv', index=False)
    df_non_login.to_csv('data/created-logs/non_login_logs_20k.csv', index=False)
    
    # Print statistics
    print("\nDataset Statistics:")
    print("="*50)
    
    print(f"\nSuccessful Logins ({len(df_success)} total):")
    print(df_success['Asset'].value_counts().head(10))
    
    print(f"\nFailed Logins ({len(df_failed)} total):")
    print(df_failed['Asset'].value_counts().head(10))
    
    print(f"\nNon-Login Events ({len(df_non_login)} total):")
    print(df_non_login['Asset'].value_counts().head(10))
    
    # Show sample logs
    print("\n" + "="*50)
    print("Sample Logs:")
    print("="*50)
    
    print("\nSuccessful Login Examples:")
    for i in range(3):
        print(f"[{i+1}] {df_success.iloc[i]['Log'][:120]}...")
    
    print("\nFailed Login Examples:")
    for i in range(3):
        print(f"[{i+1}] {df_failed.iloc[i]['Log'][:120]}...")
    
    print("\nNon-Login Examples:")
    for i in range(3):
        print(f"[{i+1}] {df_non_login.iloc[i]['Log'][:120]}...")
    
    # Create combined dataset for training
    print("\nCreating combined training dataset...")
    df_success['Label'] = 1  # login_success
    df_failed['Label'] = 2   # login_failed  
    df_non_login['Label'] = 0  # not_login
    
    df_combined = pd.concat([df_success, df_failed, df_non_login], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    df_combined.to_csv('data/created-logs/training_data_20k_combined.csv', index=False)
    print(f"Saved combined dataset with {len(df_combined)} logs to 'training_data_20k_combined.csv'")
    
    print("\nDone! You can now use these files to train your model.")

if __name__ == "__main__":
    main()