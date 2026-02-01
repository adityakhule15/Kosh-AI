import pandas as pd
import numpy as np
import re
import json
from typing import Dict, Tuple, List, Any
import traceback
from io import BytesIO
from datetime import datetime
import os

def convert_nan_to_none(obj):
    """Recursively convert NaN values to None in any data structure"""
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj) if not pd.isna(obj) else None
    elif isinstance(obj, float):
        return obj if not np.isnan(obj) else None
    elif pd.isna(obj):
        return None
    else:
        return obj

def process_statement_file(statement_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process the Statement file according to requirements:
    1. Delete rows 1 to 9 and 11
    2. Extract partner pin from descriptions
    3. Tag transactions appropriately
    
    Returns:
        - processed_df: DataFrame with processed data
        - marked_df: Original DataFrame with reconciliation marks
    """
    try:
        # Make a copy to avoid modifying the original
        original_df = statement_df.copy()
        df = statement_df.copy()
        
        # Create marked DataFrame with additional columns
        marked_df = original_df.copy()
        
        # Add reconciliation columns to marked_df
        marked_df = marked_df.astype(object)  # Convert to object type to handle mixed data
        marked_df['Row_Number_Original'] = marked_df.index + 1
        marked_df['Reconciliation_Status'] = 'Original Data'
        marked_df['Tag'] = ''
        marked_df['Pin_Number_Extracted'] = ''
        
        # Delete rows 1 to 9 and 11 (0-indexed: 0-8 and 10)
        rows_to_delete = list(range(0, 9)) + [10]
        df = df.drop(rows_to_delete).reset_index(drop=True)
        
        # Set the header row (row 0 after deletion)
        if len(df) > 0:
            # Store original headers for reference
            original_headers = df.iloc[0].tolist()
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Extract partner pin from Column D (Descriptions)
        def extract_partner_pin(description):
            if pd.isna(description):
                return None
            # Look for 9-digit number at the end of the string
            str_desc = str(description)
            matches = re.findall(r'\b(\d{9})\b', str_desc)
            return matches[-1] if matches else None
        
        df['Pin_Number'] = df['PQsTrOptOons'].apply(extract_partner_pin)
        
        # Identify duplicated transactions based on Pin Number
        df['IsDuplicate'] = df.duplicated(subset=['Pin_Number'], keep=False)
        
        # Initialize tag column
        df['Tag'] = ''
        
        # Tag "Cancel" type of duplicated transactions as "Should Reconcile"
        if 'Type' in df.columns:
            cancel_mask = (df['Type'].fillna('') == 'Cancel') & df['IsDuplicate']
            df.loc[cancel_mask, 'Tag'] = 'Should Reconcile'
        
        # Tag transactions with type as "Dollar Received" as "Should Not Reconcile"
        if 'Type' in df.columns:
            dollar_received_mask = df['Type'].fillna('') == 'Dollar received'
            df.loc[dollar_received_mask, 'Tag'] = 'Should Not Reconcile'
        
        # Tag non-duplicated transactions as "Should Reconcile"
        non_duplicate_mask = ~df['IsDuplicate'] & (df['Tag'] == '')
        df.loc[non_duplicate_mask, 'Tag'] = 'Should Reconcile'
        
        # Convert Settle.Amt to numeric (remove 'USD' and convert)
        def convert_settle_amt(value):
            if pd.isna(value):
                return np.nan
            str_val = str(value)
            # Remove currency symbols, commas, and parentheses
            str_val = str_val.replace('USD', '').replace(',', '').replace(' ', '')
            str_val = str_val.replace('(', '-').replace(')', '')
            
            try:
                return float(str_val)
            except ValueError:
                # Try to extract just the number
                num_match = re.search(r'-?\d+\.?\d*', str_val)
                return float(num_match.group()) if num_match else np.nan
        
        if 'Settle.Amt' in df.columns:
            df['Settle.Amt_Num'] = df['Settle.Amt'].apply(convert_settle_amt)
        else:
            df['Settle.Amt_Num'] = np.nan
        
        # Add original row numbers back to processed df
        df['Row_Number_Original'] = df.index + 10  # After deleting rows 1-9
        
        # Mark deleted rows
        for idx in rows_to_delete:
            if idx < len(marked_df):
                marked_df.at[idx, 'Reconciliation_Status'] = 'Deleted Row (Rows 1-9 & 11)'
        
        # Mark processed rows
        for idx, row in df.iterrows():
            original_idx = int(row['Row_Number_Original']) - 1
            if original_idx < len(marked_df):
                pin = row.get('Pin_Number', '')
                tag = row.get('Tag', '')
                
                # Store values as strings
                marked_df.at[original_idx, 'Reconciliation_Status'] = f'Processed - Pin: {pin}, Tag: {tag}'
                marked_df.at[original_idx, 'Pin_Number_Extracted'] = str(pin) if pin else ''
                marked_df.at[original_idx, 'Tag'] = str(tag) if tag else ''
        
        # Filter only rows with Pin Number for reconciliation
        df_reconcile = df[df['Pin_Number'].notna()].copy()
        df_reconcile = df_reconcile.reset_index(drop=True)
        
        return df_reconcile, marked_df
    except Exception as e:
        raise Exception(f"Error processing statement file: {str(e)}\n{traceback.format_exc()}")

def process_settlement_file(settlement_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process the Settlement file according to requirements:
    1. Delete rows 1 and 2
    2. Calculate Amount (USD) = PayoutRoundAmt ÷ APIRATE
    3. Tag transactions appropriately
    
    Returns:
        - processed_df: DataFrame with processed data
        - marked_df: Original DataFrame with reconciliation marks
    """
    try:
        # Make a copy to avoid modifying the original
        original_df = settlement_df.copy()
        df = settlement_df.copy()
        
        # Create marked DataFrame with additional columns
        marked_df = original_df.copy()
        
        # Add reconciliation columns to marked_df
        marked_df = marked_df.astype(object)  # Convert to object type to handle mixed data
        marked_df['Row_Number_Original'] = marked_df.index + 1
        marked_df['Reconciliation_Status'] = 'Original Data'
        marked_df['Tag'] = ''
        marked_df['Amount_USD_Calculated'] = ''
        
        # Delete rows 1 and 2 (0-indexed: 0 and 1)
        df = df.drop([0, 1]).reset_index(drop=True)
        
        # Set the header row (row 0 after deletion)
        if len(df) > 0:
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Clean and convert numeric columns
        def clean_numeric(value):
            if pd.isna(value):
                return np.nan
            # Remove commas and convert to float
            str_val = str(value).replace(',', '').replace(' ', '')
            try:
                return float(str_val)
            except ValueError:
                # Try to extract just the number
                num_match = re.search(r'-?\d+\.?\d*', str_val)
                return float(num_match.group()) if num_match else np.nan
        
        # Clean relevant columns
        if 'PayoutRoundAmt' in df.columns:
            df['PayoutRoundAmt_Clean'] = df['PayoutRoundAmt'].apply(clean_numeric)
        else:
            df['PayoutRoundAmt_Clean'] = np.nan
            
        if 'APIRATE' in df.columns:
            df['APIRATE_Clean'] = df['APIRATE'].apply(clean_numeric)
        else:
            df['APIRATE_Clean'] = np.nan
        
        # Calculate Amount (USD) = PayoutRoundAmt ÷ APIRATE
        df['Amount_USD'] = df['PayoutRoundAmt_Clean'] / df['APIRATE_Clean']
        
        # Round to 2 decimal places for consistency
        df['Amount_USD'] = df['Amount_USD'].round(2)
        
        # Identify duplicated transactions based on Pin Number
        df['IsDuplicate'] = df.duplicated(subset=['Pin Number'], keep=False)
        
        # Initialize tag column
        df['Tag'] = ''
        
        # Tag "Cancel" status of duplicated transactions as "Should Reconcile"
        if 'Status' in df.columns:
            # Create a boolean mask for Cancel status
            status_series = df['Status'].fillna('').astype(str)
            cancel_mask = status_series.str.contains('Cancel', case=False, na=False) & df['IsDuplicate']
            df.loc[cancel_mask, 'Tag'] = 'Should Reconcile'
        else:
            # If Status column doesn't exist, just tag based on duplicates
            cancel_mask = df['IsDuplicate']
            df.loc[cancel_mask, 'Tag'] = 'Should Reconcile'
        
        # Tag non-duplicated transactions as "Should Reconcile"
        non_duplicate_mask = ~df['IsDuplicate'] & (df['Tag'] == '')
        df.loc[non_duplicate_mask, 'Tag'] = 'Should Reconcile'
        
        # Add original row numbers back to processed df
        df['Row_Number_Original'] = df.index + 3  # After deleting rows 1-2
        
        # Mark deleted rows
        marked_df.at[0, 'Reconciliation_Status'] = 'Deleted Row (Rows 1-2)'
        marked_df.at[1, 'Reconciliation_Status'] = 'Deleted Row (Rows 1-2)'
        
        # Mark processed rows
        for idx, row in df.iterrows():
            original_idx = int(row['Row_Number_Original']) - 1
            if original_idx < len(marked_df):
                pin = row.get('Pin Number', '')
                tag = row.get('Tag', '')
                amount_usd = row.get('Amount_USD', '')
                
                # Store all values as strings
                marked_df.at[original_idx, 'Reconciliation_Status'] = f'Processed - Pin: {pin}, Tag: {tag}'
                marked_df.at[original_idx, 'Tag'] = str(tag) if tag else ''
                # Convert amount to string if it exists
                if pd.notna(amount_usd):
                    marked_df.at[original_idx, 'Amount_USD_Calculated'] = f"${amount_usd:.2f}"
                else:
                    marked_df.at[original_idx, 'Amount_USD_Calculated'] = ''
        
        # Filter only rows with Pin Number for reconciliation
        df_reconcile = df[df['Pin Number'].notna()].copy()
        df_reconcile = df_reconcile.reset_index(drop=True)
        
        return df_reconcile, marked_df
    except Exception as e:
        raise Exception(f"Error processing settlement file: {str(e)}\n{traceback.format_exc()}")

def reconcile_transactions(
    statement_df: pd.DataFrame, 
    settlement_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Reconcile transactions between Statement and Settlement files
    Returns reconciliation DataFrame and summary dictionary
    """
    try:
        # Get Should Reconcile transactions from both files
        statement_reconcile = statement_df[statement_df['Tag'] == 'Should Reconcile'].copy()
        settlement_reconcile = settlement_df[settlement_df['Tag'] == 'Should Reconcile'].copy()
        
        print(f"Statement reconcile: {len(statement_reconcile)} records")
        print(f"Settlement reconcile: {len(settlement_reconcile)} records")
        
        # Create lists to store reconciliation results
        reconciliation_results = []
        category_5 = []  # Perfect matches
        category_6 = []  # Present in both but amount mismatch
        category_7_stmt = []  # Present only in statement
        category_7_sett = []  # Present only in settlement
        
        # Get unique PartnerPins from both sets
        statement_pins = set(statement_reconcile['Pin_Number'].astype(str).tolist())
        settlement_pins = set(settlement_reconcile['Pin Number'].astype(str).tolist())
        
        print(f"Unique pins in statement: {len(statement_pins)}")
        print(f"Unique pins in settlement: {len(settlement_pins)}")
        print(f"Common pins: {len(statement_pins.intersection(settlement_pins))}")
        
        # Helper function to get transaction details
        def get_transaction_details(df, pin, source):
            try:
                # Filter by Pin Number (exact string match)
                if source == 'statement':
                    filtered_df = df[df['Pin_Number'].astype(str) == str(pin)]
                else:  # settlement
                    filtered_df = df[df['Pin Number'].astype(str) == str(pin)]
                    
                if len(filtered_df) == 0:
                    return {'date': '', 'amount': None, 'type': '', 'reference': ''}
                
                row = filtered_df.iloc[0]
                details = {
                    'date': str(row.get('Date' if source == 'statement' else 'PostDate', '')),
                    'amount': row.get('Settle.Amt_Num' if source == 'statement' else 'Amount_USD', None),
                    'type': str(row.get('Type' if source == 'statement' else 'Status', '')),
                    'reference': str(row.get('PQsTrOptOons' if source == 'statement' else 'tranno', ''))
                }
                # Convert NaN to None for JSON serialization
                for key, value in details.items():
                    if pd.isna(value):
                        details[key] = None
                return details
            except Exception as e:
                print(f"Error getting transaction details for pin {pin}: {e}")
                return {'date': '', 'amount': None, 'type': '', 'reference': ''}
        
        # Process Present in Both
        common_pins = statement_pins.intersection(settlement_pins)
        
        print(f"Processing {len(common_pins)} common pins...")
        
        for pin in common_pins:
            stmt_details = get_transaction_details(statement_reconcile, pin, 'statement')
            sett_details = get_transaction_details(settlement_reconcile, pin, 'settlement')
            
            # Calculate variance
            stmt_amount = stmt_details['amount']
            sett_amount = sett_details['amount']
            
            # Handle None/NaN values for variance calculation
            if stmt_amount is None or sett_amount is None or pd.isna(stmt_amount) or pd.isna(sett_amount):
                variance = None
                is_match = False
            else:
                variance = sett_amount - stmt_amount
                # Determine if it's a match (within tolerance)
                is_match = abs(variance) < 0.01  # Tolerance of $0.01
            
            result = {
                'Pin_Number': pin,
                'Category': 'Present in Both',
                'Statement_Date': stmt_details['date'],
                'Statement_Amount_USD': stmt_amount,
                'Statement_Type': stmt_details['type'],
                'Statement_Reference': stmt_details['reference'],
                'Settlement_Date': sett_details['date'],
                'Settlement_Amount_USD': sett_amount,
                'Settlement_Reference': sett_details['reference'],
                'Variance': variance,
                'Status': 'Matched' if is_match else 'Mismatch',
                'Classification': 'Category 5: Perfect Match' if is_match else 'Category 6: Present in Both but Amount Mismatch'
            }
            
            reconciliation_results.append(result)
            
            if is_match:
                category_5.append(result)
            else:
                category_6.append(result)
        
        # Process Present in Settlement but not in Statement
        settlement_only_pins = settlement_pins - statement_pins
        print(f"Processing {len(settlement_only_pins)} settlement-only pins...")
        
        for pin in settlement_only_pins:
            sett_details = get_transaction_details(settlement_reconcile, pin, 'settlement')
            
            result = {
                'Pin_Number': pin,
                'Category': 'Present in the Settlement File but not in the Partner Statement File',
                'Statement_Date': '',
                'Statement_Amount_USD': None,
                'Statement_Type': '',
                'Statement_Reference': '',
                'Settlement_Date': sett_details['date'],
                'Settlement_Amount_USD': sett_details['amount'],
                'Settlement_Reference': sett_details['reference'],
                'Variance': None,
                'Status': 'Unmatched',
                'Classification': 'Category 7: Present Only in Settlement File'
            }
            
            reconciliation_results.append(result)
            category_7_sett.append(result)
        
        # Process Present in Statement but not in Settlement
        statement_only_pins = statement_pins - settlement_pins
        print(f"Processing {len(statement_only_pins)} statement-only pins...")
        
        for pin in statement_only_pins:
            stmt_details = get_transaction_details(statement_reconcile, pin, 'statement')
            
            result = {
                'Pin_Number': pin,
                'Category': 'Not Present in the Settlement File but Present in the Partner Statement File',
                'Statement_Date': stmt_details['date'],
                'Statement_Amount_USD': stmt_details['amount'],
                'Statement_Type': stmt_details['type'],
                'Statement_Reference': stmt_details['reference'],
                'Settlement_Date': '',
                'Settlement_Amount_USD': None,
                'Settlement_Reference': '',
                'Variance': None,
                'Status': 'Unmatched',
                'Classification': 'Category 7: Present Only in Statement File'
            }
            
            reconciliation_results.append(result)
            category_7_stmt.append(result)
        
        # Create DataFrame from results
        reconciliation_df = pd.DataFrame(reconciliation_results)
        
        print(f"Total reconciliation results: {len(reconciliation_results)}")
        print(f"Category 5 (Perfect matches): {len(category_5)}")
        print(f"Category 6 (Mismatches): {len(category_6)}")
        print(f"Category 7 Statement only: {len(category_7_stmt)}")
        print(f"Category 7 Settlement only: {len(category_7_sett)}")
        
        # Generate summary statistics
        total_matched_amount = 0
        for item in category_5:
            if item['Statement_Amount_USD'] is not None and not pd.isna(item['Statement_Amount_USD']):
                total_matched_amount += item['Statement_Amount_USD']
        
        total_mismatch_variance = 0
        for item in category_6:
            if item['Variance'] is not None and not pd.isna(item['Variance']):
                total_mismatch_variance += item['Variance']
        
        match_rate = (len(category_5) / len(reconciliation_df) * 100) if len(reconciliation_df) > 0 else 0
        
        summary = {
            'Total Transactions Processed': len(reconciliation_df),
            'Category 5 - Perfect Matches': len(category_5),
            'Category 6 - Amount Mismatches': len(category_6),
            'Category 7 - Present Only in Statement': len(category_7_stmt),
            'Category 7 - Present Only in Settlement': len(category_7_sett),
            'Total Category 7': len(category_7_stmt) + len(category_7_sett),
            'Total Matched Amount': float(total_matched_amount) if not pd.isna(total_matched_amount) else 0.0,
            'Total Mismatch Variance': float(total_mismatch_variance) if not pd.isna(total_mismatch_variance) else 0.0,
            'Match Rate (%)': float(match_rate)
        }
        
        return reconciliation_df, summary
    except Exception as e:
        raise Exception(f"Error reconciling transactions: {str(e)}")

def create_marked_excel(
    statement_marked_df: pd.DataFrame,
    settlement_marked_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
    summary: Dict[str, Any]
) -> BytesIO:
    """
    Create an Excel file with multiple sheets:
    1. Statement File (Marked)
    2. Settlement File (Marked)
    3. Reconciliation Results
    4. Summary
    """
    try:
        output = BytesIO()
        
        # Convert DataFrames to ensure proper string formatting
        statement_export = statement_marked_df.copy()
        settlement_export = settlement_marked_df.copy()
        
        # Ensure all columns are properly formatted as strings
        for col in statement_export.columns:
            statement_export[col] = statement_export[col].astype(str)
        
        for col in settlement_export.columns:
            settlement_export[col] = settlement_export[col].astype(str)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
        
        # Write to Excel
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Statement File (Marked)
            statement_export.to_excel(writer, sheet_name='Statement_File_Marked', index=False)
            
            # Sheet 2: Settlement File (Marked)
            settlement_export.to_excel(writer, sheet_name='Settlement_File_Marked', index=False)
            
            # Sheet 3: Reconciliation Results
            reconciliation_df.to_excel(writer, sheet_name='Reconciliation_Results', index=False)
            
            # Sheet 4: Summary
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Get workbook and worksheets
            workbook = writer.book
            
            # Adjust column widths for all sheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column_cells in worksheet.columns:
                    length = max(len(str(cell.value)) for cell in column_cells)
                    worksheet.column_dimensions[column_cells[0].column_letter].width = min(length + 2, 50)
        
        output.seek(0)
        return output
    except Exception as e:
        raise Exception(f"Error creating Excel file: {str(e)}\n{traceback.format_exc()}")

def run_reconciliation(statement_path: str, settlement_path: str) -> Dict[str, Any]:
    """
    Complete reconciliation pipeline
    """
    try:
        # Read input files
        statement_raw = pd.read_excel(statement_path, sheet_name=0, header=None)
        settlement_raw = pd.read_excel(settlement_path, sheet_name=0, header=None)
        
        print(f"Statement raw shape: {statement_raw.shape}")
        print(f"Settlement raw shape: {settlement_raw.shape}")
        
        # Process files
        statement_processed, statement_marked = process_statement_file(statement_raw)
        settlement_processed, settlement_marked = process_settlement_file(settlement_raw)
        
        print(f"Statement processed shape: {statement_processed.shape}")
        print(f"Settlement processed shape: {settlement_processed.shape}")
        
        # Reconcile transactions
        reconciliation_results, summary = reconcile_transactions(
            statement_processed, 
            settlement_processed
        )
        
        print(f"Reconciliation results shape: {reconciliation_results.shape}")
        
        # Create marked Excel file
        excel_output = create_marked_excel(
            statement_marked,
            settlement_marked,
            reconciliation_results,
            summary
        )
        
        # Save Excel to a temporary file
        temp_excel_path = os.path.join('uploads', f"reconciliation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        
        # Ensure the uploads directory exists
        os.makedirs('uploads', exist_ok=True)
        
        # Write the Excel file
        with open(temp_excel_path, 'wb') as f:
            f.write(excel_output.getvalue())
        
        # Convert reconciliation results to JSON-serializable format
        reconciliation_data = []
        for _, row in reconciliation_results.iterrows():
            record = {}
            for col in reconciliation_results.columns:
                value = row[col]
                # Convert all values to JSON-serializable format
                if pd.isna(value):
                    record[col] = None
                elif isinstance(value, (np.integer, np.floating)):
                    record[col] = float(value) if not pd.isna(value) else None
                elif isinstance(value, float):
                    record[col] = value if not np.isnan(value) else None
                else:
                    record[col] = str(value) if value is not None else None
            reconciliation_data.append(record)
        
        # Calculate overall totals
        total_statement_amount = statement_processed['Settle.Amt_Num'].sum() if 'Settle.Amt_Num' in statement_processed.columns else 0
        total_settlement_amount = settlement_processed['Amount_USD'].sum() if 'Amount_USD' in settlement_processed.columns else 0
        overall_variance = total_settlement_amount - total_statement_amount
        
        # Convert NaN to None in totals
        if pd.isna(total_statement_amount):
            total_statement_amount = 0
        if pd.isna(total_settlement_amount):
            total_settlement_amount = 0
        if pd.isna(overall_variance):
            overall_variance = 0
        
        # Prepare response data
        response_data = {
            'success': True,
            'summary': convert_nan_to_none(summary),
            'reconciliation_data': reconciliation_data,
            'total_records': len(reconciliation_results),
            'overall_totals': {
                'Total Statement Amount (USD)': float(total_statement_amount),
                'Total Settlement Amount (USD)': float(total_settlement_amount),
                'Overall Variance': float(overall_variance)
            },
            'processing_metrics': {
                'statement_records_processed': len(statement_processed),
                'settlement_records_processed': len(settlement_processed),
                'reconciliation_accuracy': summary.get('Match Rate (%)', 0)
            },
            'excel_path': temp_excel_path
        }
        
        return response_data
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in reconciliation: {error_details}")
        return {
            'success': False,
            'error': f"{str(e)}",
            'report': None
        }

def allowed_file(filename):
    """Check if file has allowed extension"""
    ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS