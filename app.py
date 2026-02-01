from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import traceback
from reconciliation_logic import run_reconciliation, allowed_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'reconciliation-system-secret-key-change-in-production'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Main upload page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and processing"""
    if 'statement_file' not in request.files or 'settlement_file' not in request.files:
        return jsonify({'error': 'Both statement and settlement files are required'}), 400
    
    statement_file = request.files['statement_file']
    settlement_file = request.files['settlement_file']
    
    if statement_file.filename == '' or settlement_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not (allowed_file(statement_file.filename) and allowed_file(settlement_file.filename)):
        return jsonify({'error': 'Only Excel files (.xlsx, .xls) are allowed'}), 400
    
    statement_path = None
    settlement_path = None
    
    try:
        # Save files temporarily
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        statement_filename = f"statement_{timestamp}_{secure_filename(statement_file.filename)}"
        settlement_filename = f"settlement_{timestamp}_{secure_filename(settlement_file.filename)}"
        
        statement_path = os.path.join(app.config['UPLOAD_FOLDER'], statement_filename)
        settlement_path = os.path.join(app.config['UPLOAD_FOLDER'], settlement_filename)
        
        statement_file.save(statement_path)
        settlement_file.save(settlement_path)
        
        # Run reconciliation
        result = run_reconciliation(statement_path, settlement_path)
        
        # Clean up uploaded files
        if os.path.exists(statement_path):
            os.remove(statement_path)
        if os.path.exists(settlement_path):
            os.remove(settlement_path)
        
        if not result['success']:
            return jsonify({'error': result['error']}), 500
        
        return jsonify(result)
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Upload error: {error_details}")
        
        # Clean up any temporary files
        for filepath in [statement_path, settlement_path]:
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
        
        return jsonify({'error': f"Upload failed: {str(e)}"}), 500

@app.route('/download_excel/<filename>')
def download_excel(filename):
    """Download the marked Excel file"""
    try:
        # Secure the filename to prevent directory traversal
        safe_filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        print(f"Attempting to download file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        # Generate a download name with timestamp
        download_name = f"reconciliation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        print(f"Sending file: {file_path} as {download_name}")
        
        # Send the file
        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        error_msg = f"Error downloading file: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/cleanup_excel/<filename>', methods=['POST'])
def cleanup_excel(filename):
    """Clean up the Excel file after download"""
    try:
        safe_filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up file: {file_path}")
        
        return jsonify({'success': True, 'message': 'File cleaned up'})
    except Exception as e:
        print(f"Error cleaning up file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def show_results():
    """Display reconciliation results"""
    return render_template('results.html')

@app.route('/export/csv')
def export_csv():
    """Export results as CSV"""
    try:
        # Get data from sessionStorage (via AJAX)
        return jsonify({'error': 'Please use the Export CSV button on the results page'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup')
def cleanup():
    """Clean up temporary files"""
    try:
        import time
        import glob
        
        # Clean up any files in the uploads folder older than 1 hour
        for filepath in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
            try:
                if os.path.getmtime(filepath) < (time.time() - 3600):
                    os.remove(filepath)
                    print(f"Cleaned up old file: {filepath}")
            except:
                pass
        
        return jsonify({'success': True, 'message': 'Cleaned up temporary files'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Reconciliation System...")
    print(f"Upload folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    app.run(debug=True, host='0.0.0.0', port=5000)