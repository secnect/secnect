import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import os

from giga_dataset_gen import LogGenerator

class EnhancedDatasetGenerator(LogGenerator):
    """Enhanced dataset generator that incorporates user corrections"""
    
    def __init__(self, seed=42, corrections_file: Path = Path("feedback/corrections.json")):
        super().__init__(seed)
        self.corrections_file = corrections_file
        self.corrections_data = self.load_corrections()
    
    def load_corrections(self) -> List[Dict]:
        """Load user corrections from file"""
        try:
            if self.corrections_file.exists():
                with open(self.corrections_file, 'r') as f:
                    return json.load(f)
            return []
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load corrections file: {e}")
            return []
    
    def save_corrections(self) -> bool:
        """Save corrections back to file"""
        try:
            os.makedirs(os.path.dirname(self.corrections_file), exist_ok=True)
            with open(self.corrections_file, 'w') as f:
                json.dump(self.corrections_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving corrections: {e}")
            return False
    
    def get_unused_corrections(self) -> List[Dict]:
        """Get corrections that haven't been used for training"""
        return [record for record in self.corrections_data 
                if not record.get('used_for_training', False)]
    
    def mark_corrections_as_used(self, record_ids: List[str]) -> bool:
        """Mark specific corrections as used for training"""
        try:
            for record in self.corrections_data:
                if record.get('record_id') in record_ids:
                    record['used_for_training'] = True
                    record['training_timestamp'] = str(datetime.now())
                    record['dataset_generation_session'] = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
            return self.save_corrections()
        except Exception as e:
            print(f"Error marking corrections as used: {e}")
            return False
    
    def convert_corrections_to_training_data(self, corrections: List[Dict]) -> List[Dict]:
        """Convert corrections to training data format"""
        training_data = []
        
        for correction in corrections:
            user_correction = correction.get('user_correction', {})
            corrected_label = user_correction.get('label', '')
            
            label_mapping = {
                'successful_login': 1,
                'failed_login': 2,
                'other_login': 0,
                'not_login': 0
            }
            
            log_text = correction.get('log_text', '')
            asset_type = self.detect_asset_type(log_text)
            
            training_record = {
                'ID': correction.get('record_id', ''),
                'Asset': asset_type,
                'Log': log_text,
                'Label': label_mapping.get(corrected_label, 0),
                'Source': 'user_correction',
                'Original_Prediction': correction.get('original_prediction', {}),
                'User_Correction': user_correction,
                'Correction_Timestamp': correction.get('timestamp', '')
            }
            
            training_data.append(training_record)
        
        return training_data
    
    def detect_asset_type(self, log_text: str) -> str:
        """Detect asset type from log content (same as original)"""
        log_lower = log_text.lower()
        
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
    
    def generate_enhanced_dataset(self, 
                                total_synthetic_logs: int = 15000,
                                include_corrections: bool = True,
                                correction_boost_factor: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Generate enhanced dataset that includes user corrections
        
        Args:
            total_synthetic_logs: Number of synthetic logs to generate
            include_corrections: Whether to include user corrections
            correction_boost_factor: How many times to duplicate each correction for balanced training
            
        Returns:
            Tuple of (success_df, failed_df, non_login_df, metadata)
        """
        
        print("=" * 60)
        print("Enhanced Dataset Generation with Corrections Integration")
        print("=" * 60)
        
        unused_corrections = self.get_unused_corrections()
        corrections_to_use = []
        
        if include_corrections and unused_corrections:
            print(f"\nFound {len(unused_corrections)} unused corrections")
            
            print("\nUnused corrections available:")
            for i, correction in enumerate(unused_corrections):
                log_preview = correction.get('log_text', '')[:100] + '...' if len(correction.get('log_text', '')) > 100 else correction.get('log_text', '')
                user_correction = correction.get('user_correction', {}).get('label', 'Unknown')
                print(f"  {i+1}. {user_correction}: {log_preview}")
            

            corrections_to_use = unused_corrections
            print(f"\nUsing all {len(corrections_to_use)} corrections in dataset generation")
            
        else:
            print(f"\nNo unused corrections found or corrections disabled")
        
        corrections_training_data = []
        if corrections_to_use:
            corrections_training_data = self.convert_corrections_to_training_data(corrections_to_use)
            print(f"Converted {len(corrections_training_data)} corrections to training format")

        correction_counts = {'success': 0, 'failed': 0, 'non_login': 0}
        
        boosted_corrections = []
        for correction_data in corrections_training_data:
            label = correction_data['Label']
            for _ in range(correction_boost_factor):
                boosted_corrections.append(correction_data.copy())
            
            if label == 1:  # successful_login
                correction_counts['success'] += correction_boost_factor
            elif label == 2:  # failed_login
                correction_counts['failed'] += correction_boost_factor
            else:  # non_login
                correction_counts['non_login'] += correction_boost_factor
        
        synthetic_success = max(1, (total_synthetic_logs // 3) - correction_counts['success'])
        synthetic_failed = max(1, (total_synthetic_logs // 3) - correction_counts['failed'])
        synthetic_non_login = max(1, total_synthetic_logs - synthetic_success - synthetic_failed - 
                                 correction_counts['success'] - correction_counts['failed'] - correction_counts['non_login'])
        
        print(f"\nDataset composition:")
        print(f"  Synthetic successful logins: {synthetic_success}")
        print(f"  Synthetic failed logins: {synthetic_failed}")
        print(f"  Synthetic non-login events: {synthetic_non_login}")
        print(f"  User corrections (boosted): {len(boosted_corrections)}")
        print(f"  Total dataset size: {synthetic_success + synthetic_failed + synthetic_non_login + len(boosted_corrections)}")
        
        print(f"\nGenerating {synthetic_success + synthetic_failed + synthetic_non_login} synthetic logs...")
        df_success_synthetic, df_failed_synthetic, df_non_login_synthetic = super().generate_dataset(
            total_logs=synthetic_success + synthetic_failed + synthetic_non_login
        )
        
        df_corrections_success = pd.DataFrame()
        df_corrections_failed = pd.DataFrame()
        df_corrections_non_login = pd.DataFrame()
        
        if boosted_corrections:
            success_corrections = [c for c in boosted_corrections if c['Label'] == 1]
            failed_corrections = [c for c in boosted_corrections if c['Label'] == 2]
            non_login_corrections = [c for c in boosted_corrections if c['Label'] == 0]
            
            if success_corrections:
                df_corrections_success = pd.DataFrame(success_corrections)
                df_corrections_success['Label'] = 1
            
            if failed_corrections:
                df_corrections_failed = pd.DataFrame(failed_corrections)
                df_corrections_failed['Label'] = 2
            
            if non_login_corrections:
                df_corrections_non_login = pd.DataFrame(non_login_corrections)
                df_corrections_non_login['Label'] = 0
        
        required_columns = ['ID', 'Asset', 'Log', 'Label']

        df_success_synthetic['Label'] = 1
        df_failed_synthetic['Label'] = 2
        df_non_login_synthetic['Label'] = 0
        
        if not df_corrections_success.empty:
            # Ensure corrections have all required columns
            for col in required_columns:
                if col not in df_corrections_success.columns:
                    if col == 'ID':
                        df_corrections_success[col] = range(len(df_success_synthetic) + 1, 
                                                          len(df_success_synthetic) + len(df_corrections_success) + 1)
            df_success_final = pd.concat([df_success_synthetic[required_columns + ['Source'] if 'Source' in df_success_synthetic.columns else required_columns], 
                                        df_corrections_success[required_columns + ['Source']]], ignore_index=True)
        else:
            df_success_final = df_success_synthetic.copy()
            df_success_final['Source'] = 'synthetic'
        
        if not df_corrections_failed.empty:
            for col in required_columns:
                if col not in df_corrections_failed.columns:
                    if col == 'ID':
                        df_corrections_failed[col] = range(len(df_failed_synthetic) + 1, 
                                                         len(df_failed_synthetic) + len(df_corrections_failed) + 1)
            df_failed_final = pd.concat([df_failed_synthetic[required_columns + ['Source'] if 'Source' in df_failed_synthetic.columns else required_columns], 
                                       df_corrections_failed[required_columns + ['Source']]], ignore_index=True)
        else:
            df_failed_final = df_failed_synthetic.copy()
            df_failed_final['Source'] = 'synthetic'
        
        if not df_corrections_non_login.empty:
            for col in required_columns:
                if col not in df_corrections_non_login.columns:
                    if col == 'ID':
                        df_corrections_non_login[col] = range(len(df_non_login_synthetic) + 1, 
                                                            len(df_non_login_synthetic) + len(df_corrections_non_login) + 1)
            df_non_login_final = pd.concat([df_non_login_synthetic[required_columns + ['Source'] if 'Source' in df_non_login_synthetic.columns else required_columns], 
                                          df_corrections_non_login[required_columns + ['Source']]], ignore_index=True)
        else:
            df_non_login_final = df_non_login_synthetic.copy()
            df_non_login_final['Source'] = 'synthetic'
        
        if corrections_to_use:
            record_ids = [c.get('record_id') for c in corrections_to_use]
            if self.mark_corrections_as_used(record_ids):
                print(f"\nMarked {len(record_ids)} corrections as used for training")
            else:
                print(f"\nWarning: Could not mark corrections as used")
        
        metadata = {
            'generation_timestamp': str(datetime.now()),
            'total_logs': len(df_success_final) + len(df_failed_final) + len(df_non_login_final),
            'synthetic_logs': synthetic_success + synthetic_failed + synthetic_non_login,
            'correction_logs': len(boosted_corrections),
            'corrections_used': len(corrections_to_use),
            'correction_boost_factor': correction_boost_factor,
            'successful_logins': len(df_success_final),
            'failed_logins': len(df_failed_final),
            'non_login_events': len(df_non_login_final),
            'corrections_by_type': correction_counts,
            'used_correction_ids': [c.get('record_id') for c in corrections_to_use]
        }
        
        print(f"\nDataset generation complete!")
        print(f"Final dataset size: {metadata['total_logs']} logs")
        print(f"  - Successful logins: {metadata['successful_logins']}")
        print(f"  - Failed logins: {metadata['failed_logins']}")
        print(f"  - Non-login events: {metadata['non_login_events']}")
        
        return df_success_final, df_failed_final, df_non_login_final, metadata
    
    def save_enhanced_dataset(self, 
                            df_success: pd.DataFrame, 
                            df_failed: pd.DataFrame, 
                            df_non_login: pd.DataFrame,
                            metadata: Dict,
                            output_dir: str = 'data/enhanced-logs') -> Dict[str, str]:
        """Save enhanced dataset with metadata"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        files = {
            'successful_logins': f'successful_login_logs_enhanced_{timestamp}.csv',
            'failed_logins': f'failed_login_logs_enhanced_{timestamp}.csv',
            'non_login_events': f'non_login_logs_enhanced_{timestamp}.csv',
            'combined': f'training_data_enhanced_combined_{timestamp}.csv',
            'metadata': f'dataset_metadata_{timestamp}.json'
        }
        
        df_success.to_csv(output_path / files['successful_logins'], index=False)
        df_failed.to_csv(output_path / files['failed_logins'], index=False)
        df_non_login.to_csv(output_path / files['non_login_events'], index=False)
        
        df_combined = pd.concat([df_success, df_failed, df_non_login], ignore_index=True)
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        df_combined.to_csv(output_path / files['combined'], index=False)
        
        with open(output_path / files['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        file_paths = {key: str(output_path / filename) for key, filename in files.items()}
        
        print(f"\nSaved enhanced dataset to: {output_path}")
        for key, path in file_paths.items():
            print(f"  {key}: {Path(path).name}")
        
        return file_paths

def main():
    """Generate enhanced dataset with corrections integration"""
    
    print("Enhanced Dataset Generator with Corrections Integration")
    print("=" * 60)
    
    generator = EnhancedDatasetGenerator(seed=42)
    
    total_synthetic_logs = 15000  # Reduced to make room for corrections
    include_corrections = True
    correction_boost_factor = 5  # Boost corrections to balance dataset
    
    print(f"Configuration:")
    print(f"  Synthetic logs to generate: {total_synthetic_logs}")
    print(f"  Include corrections: {include_corrections}")
    print(f"  Correction boost factor: {correction_boost_factor}")
    
    df_success, df_failed, df_non_login, metadata = generator.generate_enhanced_dataset(
        total_synthetic_logs=total_synthetic_logs,
        include_corrections=include_corrections,
        correction_boost_factor=correction_boost_factor
    )
    
    file_paths = generator.save_enhanced_dataset(df_success, df_failed, df_non_login, metadata)
    
    print("\n" + "=" * 60)
    print("Enhanced Dataset Statistics:")
    print("=" * 60)
    
    print(f"\nSuccessful Logins ({len(df_success)} total):")
    if 'Source' in df_success.columns:
        print("By source:")
        print(df_success['Source'].value_counts())
    print("By asset type:")
    print(df_success['Asset'].value_counts().head(10))
    
    print(f"\nFailed Logins ({len(df_failed)} total):")
    if 'Source' in df_failed.columns:
        print("By source:")
        print(df_failed['Source'].value_counts())
    print("By asset type:")
    print(df_failed['Asset'].value_counts().head(10))
    
    print(f"\nNon-Login Events ({len(df_non_login)} total):")
    if 'Source' in df_non_login.columns:
        print("By source:")
        print(df_non_login['Source'].value_counts())
    print("By asset type:")
    print(df_non_login['Asset'].value_counts().head(10))

    print("\n" + "=" * 60)
    print("Sample Enhanced Logs:")
    print("=" * 60)
    
    if 'Source' in df_success.columns:
        user_corrections = df_success[df_success['Source'] == 'user_correction']
        if not user_corrections.empty:
            print(f"\nUser Correction Examples (Successful Logins):")
            for i in range(min(2, len(user_corrections))):
                log_text = user_corrections.iloc[i]['Log']
                print(f"[{i+1}] {log_text[:120]}...")
    
    if 'Source' in df_failed.columns:
        user_corrections = df_failed[df_failed['Source'] == 'user_correction']
        if not user_corrections.empty:
            print(f"\nUser Correction Examples (Failed Logins):")
            for i in range(min(2, len(user_corrections))):
                log_text = user_corrections.iloc[i]['Log']
                print(f"[{i+1}] {log_text[:120]}...")
    
    print(f"\nSynthetic Examples:")
    print("Successful Login:")
    print(f"[1] {df_success[df_success.get('Source', 'synthetic') == 'synthetic'].iloc[0]['Log'][:120]}...")
    print("Failed Login:")
    print(f"[1] {df_failed[df_failed.get('Source', 'synthetic') == 'synthetic'].iloc[0]['Log'][:120]}...")
    print("Non-Login:")
    print(f"[1] {df_non_login[df_non_login.get('Source', 'synthetic') == 'synthetic'].iloc[0]['Log'][:120]}...")
    
    print(f"\nDataset files saved to: {Path(file_paths['combined']).parent}")
    print(f"Ready for model training!")

if __name__ == "__main__":
    main()