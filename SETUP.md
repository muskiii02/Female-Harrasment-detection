# Quick Setup Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Verify Model File

Make sure `best_harassment_detector.keras` is in the project root directory.

## Step 3: Run the Application

```bash
python app.py
```

This will:
- Create the SQLite database
- Create a default admin account
- Start the Flask server on http://localhost:5000

## Step 4: Login

**Default Admin Account:**
- Username: `admin`
- Password: `admin123`

⚠️ **IMPORTANT: Change this password immediately after first login!**

## Step 5: Test the System

### As a Regular User:
1. Register a new account
2. Login
3. Upload an image
4. View detection results

### As Admin:
1. Login with admin account
2. View dashboard statistics
3. Check alerts for harassment detections
4. Review and manage alerts
5. View all uploads and users

## Troubleshooting

### Model Not Loading
- Ensure `best_harassment_detector.keras` is in the project root
- Check that TensorFlow/Keras is installed correctly

### Database Errors
- Delete `harassment_detection.db` and restart the app to recreate it

### Port Already in Use
- Change the port in `app.py`: `app.run(debug=True, host='0.0.0.0', port=5001)`

## Features Overview

### User Features:
- ✅ Registration and Login
- ✅ Upload images for harassment detection  
- ✅ View upload history
- ✅ See personal statistics

### Admin Features:
- ✅ System-wide dashboard
- ✅ Alert management system
- ✅ Review harassment detections
- ✅ View all users and uploads
- ✅ Add admin notes to alerts

## Database Schema

- **Users**: id, username, email, password_hash, is_admin, created_at
- **Uploads**: id, user_id, filename, filepath, detection_result, confidence, uploaded_at, reviewed
- **Alerts**: id, upload_id, status, admin_notes, created_at, reviewed_at, reviewed_by

All relationships are properly configured with foreign keys and cascade deletes.

