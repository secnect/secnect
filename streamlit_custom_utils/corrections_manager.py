from typing import List, Dict, Any
from pathlib import Path
import streamlit as st
import pandas as pd
import datetime
import hashlib
import json
import os

# Feedback system setup
FEEDBACK_FILE = Path("feedback") / "secbert_feedback.json"
CORRECTIONS_FILE = Path("feedback") / "corrections.json"

class CorrectionsManager:
    """Manages corrections data and training tracking"""
    
    def __init__(self, corrections_file: Path = CORRECTIONS_FILE):
        self.corrections_file = corrections_file
        self.corrections_data = self.load_corrections()
    
    def load_corrections(self) -> List[Dict]:
        """Load corrections from JSON file"""
        try:
            if self.corrections_file.exists():
                with open(self.corrections_file, 'r') as f:
                    data = json.load(f)
                # Ensure each record has required fields
                for record in data:
                    if 'used_for_training' not in record:
                        record['used_for_training'] = False
                    if 'record_id' not in record:
                        record['record_id'] = hashlib.md5(
                            (record.get('log_text', '') + str(record.get('timestamp', ''))).encode()
                        ).hexdigest()[:12]
                return data
            return []
        except (FileNotFoundError, json.JSONDecodeError) as e:
            st.error(f"Error loading corrections file: {e}")
            return []
    
    def save_corrections(self) -> bool:
        """Save corrections data back to file"""
        try:
            os.makedirs(os.path.dirname(self.corrections_file), exist_ok=True)
            with open(self.corrections_file, 'w') as f:
                json.dump(self.corrections_data, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving corrections: {e}")
            return False
    
    def get_corrections_df(self) -> pd.DataFrame:
        """Convert corrections to DataFrame for display"""
        if not self.corrections_data:
            return pd.DataFrame(columns=[
                'Record ID', 'Log Text', 'Original Prediction', 'User Correction', 
                'Timestamp', 'Used for Training'
            ])
        
        df_data = []
        for record in self.corrections_data:
            df_data.append({
                'Record ID': record.get('record_id', 'N/A'),
                'Log Text': record.get('log_text', '')[:100] + '...' if len(record.get('log_text', '')) > 100 else record.get('log_text', ''),
                'Full Log Text': record.get('log_text', ''),
                'Original Prediction': self._format_prediction(record.get('original_prediction', {})),
                'User Correction': self._format_prediction(record.get('user_correction', {})),
                'Timestamp': record.get('timestamp', 'N/A'),
                'Used for Training': record.get('used_for_training', False)
            })
        
        return pd.DataFrame(df_data)
    
    def _format_prediction(self, prediction: Dict) -> str:
        """Format prediction for display"""
        if not prediction:
            return "N/A"
        label = prediction.get('label', 'Unknown')
        confidence = prediction.get('confidence', 0)
        return f"{label} ({confidence:.3f})"
    
    def mark_as_used(self, record_ids: List[str]) -> bool:
        """Mark specific records as used for training"""
        try:
            for record in self.corrections_data:
                if record.get('record_id') in record_ids:
                    record['used_for_training'] = True
                    record['training_timestamp'] = str(datetime.now())
            return self.save_corrections()
        except Exception as e:
            st.error(f"Error marking records as used: {e}")
            return False
    
    def get_unused_corrections(self) -> List[Dict]:
        """Get corrections that haven't been used for training"""
        return [record for record in self.corrections_data if not record.get('used_for_training', False)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about corrections"""
        total = len(self.corrections_data)
        used = sum(1 for record in self.corrections_data if record.get('used_for_training', False))
        unused = total - used
        
        # Count by correction type
        correction_types = {}
        for record in self.corrections_data:
            correction = record.get('user_correction', {})
            label = correction.get('label', 'Unknown')
            correction_types[label] = correction_types.get(label, 0) + 1
        
        return {
            'total': total,
            'used': used,
            'unused': unused,
            'correction_types': correction_types
        }

def show_corrections_management():
    """Display corrections management interface"""
    st.header("Corrections Management")
    st.markdown("Manage user corrections and track their usage in training datasets.")
    
    # Initialize corrections manager
    corrections_manager = CorrectionsManager()
    
    # Show statistics
    stats = corrections_manager.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Corrections", stats['total'])
    with col2:
        st.metric("Used for Training", stats['used'], delta=f"{stats['used']/max(stats['total'], 1)*100:.1f}%")
    with col3:
        st.metric("Unused", stats['unused'])
    with col4:
        if stats['total'] > 0:
            st.metric("Usage Rate", f"{stats['used']/stats['total']*100:.1f}%")
        else:
            st.metric("Usage Rate", "0%")
    
    # Show correction types distribution
    if stats['correction_types']:
        st.subheader("📊 Correction Types Distribution")
        correction_df = pd.DataFrame(list(stats['correction_types'].items()), 
                                   columns=['Correction Type', 'Count'])
        st.bar_chart(correction_df.set_index('Correction Type'))
    
    # Display corrections table
    st.subheader("📋 All Corrections")
    
    corrections_df = corrections_manager.get_corrections_df()
    
    if corrections_df.empty:
        st.info("No corrections found. Start analyzing logs to generate corrections.")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_used = st.selectbox(
            "Filter by training status:",
            ["All", "Used for Training", "Not Used for Training"]
        )
    
    with col2:
        search_text = st.text_input("Search in log text:", placeholder="Enter search term...")
    
    # Apply filters
    filtered_df = corrections_df.copy()
    
    if filter_used == "Used for Training":
        filtered_df = filtered_df[filtered_df['Used for Training'] == True]
    elif filter_used == "Not Used for Training":
        filtered_df = filtered_df[filtered_df['Used for Training'] == False]
    
    if search_text:
        filtered_df = filtered_df[
            filtered_df['Full Log Text'].str.contains(search_text, case=False, na=False)
        ]
    
    st.write(f"Showing {len(filtered_df)} of {len(corrections_df)} corrections")
    
    # Display table with selection
    if len(filtered_df) > 0:
        # Add selection column
        selection_col = st.columns([0.1, 0.9])
        
        with selection_col[0]:
            select_all = st.checkbox("Select All")
        
        with selection_col[1]:
            if st.button("Mark Selected as Used for Training", disabled=len(filtered_df) == 0):
                selected_records = []
                if select_all:
                    selected_records = filtered_df['Record ID'].tolist()
                else:
                    # Get selected records from checkboxes
                    for idx, row in filtered_df.iterrows():
                        if st.session_state.get(f"select_{row['Record ID']}", False):
                            selected_records.append(row['Record ID'])
                
                if selected_records:
                    if corrections_manager.mark_as_used(selected_records):
                        st.success(f"Marked {len(selected_records)} records as used for training!")
                        st.rerun()
                    else:
                        st.error("Failed to update records.")
                else:
                    st.warning("No records selected.")
        
        # Display detailed table
        for idx, row in filtered_df.iterrows():
            with st.expander(f"Record {row['Record ID']} - {row['Original Prediction']} → {row['User Correction']}"):
                
                # Selection checkbox (if not used for training)
                if not row['Used for Training']:
                    st.checkbox(
                        f"Select for training", 
                        key=f"select_{row['Record ID']}",
                        value=select_all
                    )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Log Text:**")
                    st.text_area("", value=row['Full Log Text'], height=100, disabled=True, key=f"log_{row['Record ID']}")
                    st.markdown(f"**Timestamp:** {row['Timestamp']}")
                
                with col2:
                    st.markdown("**Original Prediction:**")
                    st.write(row['Original Prediction'])
                    st.markdown("**User Correction:**")
                    st.write(row['User Correction'])
                    st.markdown(f"**Training Status:** {'✅ Used' if row['Used for Training'] else '❌ Not Used'}")
                
                if row['Used for Training']:
                    st.success("This correction has been used for training.")
    
    # Export functionality
    st.subheader("📤 Export Corrections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export All Corrections (CSV)"):
            csv = corrections_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"corrections_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        unused_corrections = corrections_manager.get_unused_corrections()
        if unused_corrections:
            if st.button("Export Unused for Training (JSON)"):
                json_data = json.dumps(unused_corrections, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"unused_corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
