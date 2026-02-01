import os
import json
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse, FileResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .reconciliation_logic import run_reconciliation, allowed_file
import traceback

def index(request):
    """Main upload page"""
    return render(request, 'reconciliation_app/index.html')

@csrf_exempt
def upload_files(request):
    """Handle file upload and processing"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    if 'statement_file' not in request.FILES or 'settlement_file' not in request.FILES:
        return JsonResponse({'error': 'Both statement and settlement files are required'}, status=400)
    
    statement_file = request.FILES['statement_file']
    settlement_file = request.FILES['settlement_file']
    
    if statement_file.name == '' or settlement_file.name == '':
        return JsonResponse({'error': 'No selected file'}, status=400)
    
    if not (allowed_file(statement_file.name) and allowed_file(settlement_file.name)):
        return JsonResponse({'error': 'Only Excel files (.xlsx, .xls) are allowed'}, status=400)
    
    statement_path = None
    settlement_path = None
    
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(settings.MEDIA_ROOT)
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save files temporarily
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        statement_filename = f"statement_{timestamp}_{statement_file.name}"
        settlement_filename = f"settlement_{timestamp}_{settlement_file.name}"
        
        statement_path = os.path.join(uploads_dir, statement_filename)
        settlement_path = os.path.join(uploads_dir, settlement_filename)
        
        # Save uploaded files
        with open(statement_path, 'wb+') as destination:
            for chunk in statement_file.chunks():
                destination.write(chunk)
        
        with open(settlement_path, 'wb+') as destination:
            for chunk in settlement_file.chunks():
                destination.write(chunk)
        
        # Run reconciliation
        result = run_reconciliation(statement_path, settlement_path)
        
        # Clean up uploaded files
        if os.path.exists(statement_path):
            os.remove(statement_path)
        if os.path.exists(settlement_path):
            os.remove(settlement_path)
        
        if not result['success']:
            return JsonResponse({'error': result['error']}, status=500)
        
        return JsonResponse(result)
        
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
        
        return JsonResponse({'error': f"Upload failed: {str(e)}"}, status=500)

def download_excel(request, filename):
    """Download the marked Excel file"""
    try:
        # Secure the filename
        file_path = os.path.join(filename)
        
        print(f"Attempting to download file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return JsonResponse({'error': 'File not found'}, status=404)
        
        # Generate a download name with timestamp
        download_name = f"reconciliation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        print(f"Sending file: {file_path} as {download_name}")
        
        # Send the file
        response = FileResponse(
            open(file_path, 'rb'),
            as_attachment=True,
            filename=download_name,
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
        return response
        
    except Exception as e:
        error_msg = f"Error downloading file: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return JsonResponse({'error': error_msg}, status=500)

@csrf_exempt
def cleanup_excel(request, filename):
    """Clean up the Excel file after download"""
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up file: {file_path}")
        
        return JsonResponse({'success': True, 'message': 'File cleaned up'})
    except Exception as e:
        print(f"Error cleaning up file: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

def show_results(request):
    """Display reconciliation results"""
    return render(request, 'reconciliation_app/results.html')

@csrf_exempt
def cleanup(request):
    """Clean up temporary files"""
    try:
        import time
        import glob
        
        # Clean up any files in the uploads folder older than 1 hour
        for filepath in glob.glob(os.path.join(settings.MEDIA_ROOT, '*')):
            try:
                if os.path.getmtime(filepath) < (time.time() - 3600):
                    os.remove(filepath)
                    print(f"Cleaned up old file: {filepath}")
            except:
                pass
        
        return JsonResponse({'success': True, 'message': 'Cleaned up temporary files'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def health_check(request):
    """Health check endpoint"""
    return JsonResponse({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

    