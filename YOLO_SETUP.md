# YOLO Integration Setup

## Overview
The system now uses YOLOv8 for person detection before running harassment detection. This ensures that only images containing people are analyzed.

## Installation

### 1. Install New Dependencies
```bash
pip install ultralytics opencv-python
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. YOLO Model Download
When you first run the application, YOLOv8 will automatically download the `yolov8n.pt` (nano) model (~6MB). This happens automatically on first use.

## How It Works

### Two-Stage Detection Pipeline:

#### Stage 1: Person Detection (YOLO)
- Uses YOLOv8 nano model for fast person detection
- Checks if at least one person is present in the image
- Returns person count and confidence

#### Stage 2: Harassment Detection (Keras)
- Only runs if person(s) detected in Stage 1
- Uses your trained ResNet model for harassment classification
- Returns harassment/normal classification with confidence

### User Experience:

**First 4 Uploads (Demo Mode):**
1. **1st Upload** → Person check → Harassment (85% confidence)
2. **2nd Upload** → Person check → Harassment (90% confidence)
3. **3rd Upload** → Person check → Normal (92% confidence)
4. **4th Upload** → Person check → Normal (87% confidence)

**5th Upload Onwards:**
- Full YOLO + Keras pipeline
- Real model predictions

**If No Person Detected:**
- Shows warning message: "No Human Detected"
- Image is NOT saved to database
- User can upload another image

## Technical Details

### YOLO Configuration
- Model: YOLOv8n (nano - fastest, smallest)
- Person class: 0 (COCO dataset)
- Confidence threshold: 0.5
- Detection speed: ~50ms per image on CPU

### Error Handling
- No person detected → User-friendly error message
- YOLO model not loaded → Graceful fallback
- Keras model not loaded → Error with explanation

## Benefits

✅ **Relevant Images Only**: Only analyzes images with people  
✅ **Fast Pre-screening**: YOLO is very fast (~50ms)  
✅ **Better Accuracy**: Reduces false positives from non-human images  
✅ **User Feedback**: Clear message when no person detected  
✅ **Resource Efficient**: Doesn't waste resources on irrelevant images

## Testing

Upload these types of images to test:
1. **Images with people** → Should proceed to harassment detection
2. **Landscape/object photos** → Should show "No Human Detected"
3. **Animal photos** → Should show "No Human Detected"
4. **Group photos** → Should detect multiple persons

## Troubleshooting

**Issue**: YOLO model download fails  
**Solution**: Download manually from https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

**Issue**: "No human detected" for images with people  
**Solution**: Check image quality, lighting, and person visibility. Adjust confidence threshold in `app.py` (line ~75)

**Issue**: Slow performance  
**Solution**: YOLOv8n is already the fastest. Consider GPU if available.

