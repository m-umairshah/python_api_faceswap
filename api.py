from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import io
import base64
from PIL import Image
import face_recognition
import math
from datetime import datetime
import json
import requests
import tempfile
import uuid

# Create Flask API app
api_app = Flask(__name__)
api_app.secret_key = 'your-api-secret-key-here'

# Enable CORS for cross-origin requests
CORS(api_app, origins=['*'])

# Configuration
API_UPLOAD_FOLDER = 'api_uploads'
API_PROCESSED_FOLDER = 'api_processed'
API_FACES_FOLDER = 'api_extracted_faces'
API_ANALYSIS_FOLDER = 'api_analysis_results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

def ensure_api_directories():
    """Ensure all required API directories exist"""
    directories = [API_UPLOAD_FOLDER, API_PROCESSED_FOLDER, API_FACES_FOLDER, API_ANALYSIS_FOLDER]
    for folder in directories:
        try:
            os.makedirs(folder, exist_ok=True)
            print(f"API Directory ensured: {os.path.abspath(folder)}")
        except Exception as e:
            print(f"Warning: Could not create API directory {folder}: {str(e)}")

# Create directories at startup
ensure_api_directories()

api_app.config['API_UPLOAD_FOLDER'] = API_UPLOAD_FOLDER
api_app.config['API_PROCESSED_FOLDER'] = API_PROCESSED_FOLDER
api_app.config['API_FACES_FOLDER'] = API_FACES_FOLDER
api_app.config['API_ANALYSIS_FOLDER'] = API_ANALYSIS_FOLDER
api_app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def safe_path_join(*args):
    """Safely join path components using os.path.join"""
    return os.path.normpath(os.path.join(*args))

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    try:
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    except (IndexError, TypeError):
        return 0

def safe_get_landmark(landmarks, index, default=(0, 0)):
    """Safely get landmark point with bounds checking"""
    try:
        if 0 <= index < len(landmarks):
            return landmarks[index]
        return default
    except (IndexError, TypeError):
        return default

def draw_facial_mesh(image, landmarks, face_left=0, face_top=0):
    """Draw detailed facial mesh like in the reference image"""
    try:
        # Convert landmarks to face-relative coordinates if needed
        if face_left != 0 or face_top != 0:
            adjusted_landmarks = []
            for point in landmarks:
                adjusted_landmarks.append((point[0] - face_left, point[1] - face_top))
            landmarks = adjusted_landmarks
        
        # Define facial mesh connections (based on 68-point facial landmarks)
        # Face outline connections
        face_outline = list(range(0, 17))
        
        # Eyebrow connections
        left_eyebrow = list(range(17, 22))
        right_eyebrow = list(range(22, 27))
        
        # Nose bridge connections
        nose_bridge = list(range(27, 31))
        nose_bottom = list(range(31, 36))
        
        # Eye connections
        left_eye = list(range(36, 42)) + [36]  # Close the loop
        right_eye = list(range(42, 48)) + [42]  # Close the loop
        
        # Mouth connections
        outer_mouth = list(range(48, 60)) + [48]  # Close the loop
        inner_mouth = list(range(60, 68)) + [60]  # Close the loop
        
        # Draw white mesh lines for face structure
        mesh_color = (255, 255, 255)  # White
        mesh_thickness = 1
        
        # Draw face outline
        for i in range(len(face_outline) - 1):
            if face_outline[i] < len(landmarks) and face_outline[i + 1] < len(landmarks):
                pt1 = tuple(map(int, landmarks[face_outline[i]]))
                pt2 = tuple(map(int, landmarks[face_outline[i + 1]]))
                cv2.line(image, pt1, pt2, mesh_color, mesh_thickness)
        
        # Draw eyebrows
        for eyebrow in [left_eyebrow, right_eyebrow]:
            for i in range(len(eyebrow) - 1):
                if eyebrow[i] < len(landmarks) and eyebrow[i + 1] < len(landmarks):
                    pt1 = tuple(map(int, landmarks[eyebrow[i]]))
                    pt2 = tuple(map(int, landmarks[eyebrow[i + 1]]))
                    cv2.line(image, pt1, pt2, mesh_color, mesh_thickness)
        
        # Draw nose bridge
        for i in range(len(nose_bridge) - 1):
            if nose_bridge[i] < len(landmarks) and nose_bridge[i + 1] < len(landmarks):
                pt1 = tuple(map(int, landmarks[nose_bridge[i]]))
                pt2 = tuple(map(int, landmarks[nose_bridge[i + 1]]))
                cv2.line(image, pt1, pt2, mesh_color, mesh_thickness)
        
        # Draw nose bottom
        for i in range(len(nose_bottom) - 1):
            if nose_bottom[i] < len(landmarks) and nose_bottom[i + 1] < len(landmarks):
                pt1 = tuple(map(int, landmarks[nose_bottom[i]]))
                pt2 = tuple(map(int, landmarks[nose_bottom[i + 1]]))
                cv2.line(image, pt1, pt2, mesh_color, mesh_thickness)
        
        # Draw additional mesh lines for face structure
        # Vertical lines
        if len(landmarks) >= 68:
            # Center line from forehead to chin
            center_points = [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]
            for i in range(len(center_points) - 1):
                if center_points[i] < len(landmarks) and center_points[i + 1] < len(landmarks):
                    pt1 = tuple(map(int, landmarks[center_points[i]]))
                    pt2 = tuple(map(int, landmarks[center_points[i + 1]]))
                    cv2.line(image, pt1, pt2, mesh_color, mesh_thickness)
            
            # Horizontal lines across face
            horizontal_connections = [
                (0, 16),   # Jaw line
                (1, 15),   # Upper jaw
                (2, 14),   # Mid jaw
                (17, 26),  # Eyebrow line
                (36, 45),  # Eye line
                (31, 35),  # Nose base
                (48, 54),  # Mouth line
            ]
            
            for pt1_idx, pt2_idx in horizontal_connections:
                if pt1_idx < len(landmarks) and pt2_idx < len(landmarks):
                    pt1 = tuple(map(int, landmarks[pt1_idx]))
                    pt2 = tuple(map(int, landmarks[pt2_idx]))
                    cv2.line(image, pt1, pt2, mesh_color, mesh_thickness)
            
            # Additional diagonal and cross lines for full mesh
            diagonal_connections = [
                (0, 8),    # Left jaw to chin
                (16, 8),   # Right jaw to chin
                (17, 21),  # Left eyebrow diagonal
                (22, 26),  # Right eyebrow diagonal
                (36, 39),  # Left eye diagonal
                (42, 45),  # Right eye diagonal
                (31, 33),  # Nose diagonal
                (48, 51),  # Mouth diagonal left
                (51, 54),  # Mouth diagonal right
            ]
            
            for pt1_idx, pt2_idx in diagonal_connections:
                if pt1_idx < len(landmarks) and pt2_idx < len(landmarks):
                    pt1 = tuple(map(int, landmarks[pt1_idx]))
                    pt2 = tuple(map(int, landmarks[pt2_idx]))
                    cv2.line(image, pt1, pt2, mesh_color, mesh_thickness)
        
        # Draw colored feature highlights
        feature_thickness = 2
        
        # Draw eyes with green color
        eye_color = (255, 255, 255)  # Green
        for eye_points in [left_eye[:-1], right_eye[:-1]]:  # Remove duplicate point
            for i in range(len(eye_points)):
                if eye_points[i] < len(landmarks) and eye_points[(i + 1) % len(eye_points)] < len(landmarks):
                    pt1 = tuple(map(int, landmarks[eye_points[i]]))
                    pt2 = tuple(map(int, landmarks[eye_points[(i + 1) % len(eye_points)]]))
                    cv2.line(image, pt1, pt2, eye_color, feature_thickness)
        
        # Draw mouth with red color
        mouth_color = (0, 0, 255)  # Red
        for mouth_points in [outer_mouth[:-1], inner_mouth[:-1]]:  # Remove duplicate point
            for i in range(len(mouth_points)):
                if mouth_points[i] < len(landmarks) and mouth_points[(i + 1) % len(mouth_points)] < len(landmarks):
                    pt1 = tuple(map(int, landmarks[mouth_points[i]]))
                    pt2 = tuple(map(int, landmarks[mouth_points[(i + 1) % len(mouth_points)]]))
                    cv2.line(image, pt1, pt2, mouth_color, feature_thickness)
        
        # Draw landmark points
        point_color = (255, 255, 255)  # White points
        point_radius = 1
        
        for i, point in enumerate(landmarks):
            if i < 68:  # Only draw the 68 standard landmarks
                center = tuple(map(int, point))
                cv2.circle(image, center, point_radius, point_color, -1)
        
        return image
        
    except Exception as e:
        print(f"Error drawing facial mesh: {str(e)}")
        return image

def image_to_base64(image):
    """Convert OpenCV image to base64 string for web display"""
    try:
        if image is None:
            return None
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return None

def create_feature_highlighted_images_with_mesh(face_region, face_landmarks, face_left, face_top):
    """Create separate images with individual features highlighted and mesh overlay"""
    feature_images = {}
    
    # Define feature groups and their colors
    feature_groups = {
        'nose': {
            'features': ['nose_bridge', 'nose_tip'],
            'color': (255, 255, 255),  # Red
            'thickness': 2
        },
        'lips': {
            'features': ['top_lip', 'bottom_lip'],
            'color': (255, 255, 255),  # Magenta
            'thickness': 2
        },
        'eyes': {
            'features': ['left_eye', 'right_eye'],
            'color': (255, 255, 255),  # Green
            'thickness': 2
        },
        'eyebrows': {
            'features': ['left_eyebrow', 'right_eyebrow'],
            'color': (255, 255, 255),  # Cyan
            'thickness': 2
        },
        'face_outline': {
            'features': ['chin', 'left_eyebrow', 'right_eyebrow'],
            'color': (255, 255, 255),  # Yellow
            'thickness': 2
        }
    }
    
    try:
        # Convert landmarks to face-relative coordinates
        face_landmarks_adjusted = {}
        for feature_name, points in face_landmarks.items():
            face_points = []
            for point in points:
                face_x = point[0] - face_left
                face_y = point[1] - face_top
                if 0 <= face_x < face_region.shape[1] and 0 <= face_y < face_region.shape[0]:
                    face_points.append((face_x, face_y))
            if face_points:
                face_landmarks_adjusted[feature_name] = face_points
        
        # Convert to landmarks_points format for mesh drawing
        landmarks_points = []
        for feature_name, points in face_landmarks_adjusted.items():
            landmarks_points.extend(points)
        
        for group_name, group_config in feature_groups.items():
            # Create a copy of the original face for this feature
            highlighted_face = face_region.copy()
            
            # First draw the facial mesh
            if landmarks_points:
                draw_facial_mesh(highlighted_face, landmarks_points)
            
            # Then highlight the specific features
            for feature_name in group_config['features']:
                if feature_name in face_landmarks_adjusted:
                    points = face_landmarks_adjusted[feature_name]
                    
                    if len(points) > 2:
                        points_array = np.array(points, dtype=np.int32)
                        
                        # Draw thick outline for the feature
                        cv2.polylines(highlighted_face, [points_array], 
                                    feature_name in ['top_lip', 'bottom_lip', 'left_eye', 'right_eye'], 
                                    group_config['color'], group_config['thickness'])
                        
                        # Add feature points
                        for point in points:
                            cv2.circle(highlighted_face, tuple(map(int, point)), 3, group_config['color'], -1)
                        
                        # Add feature filling for lips and eyes
                        if feature_name in ['top_lip', 'bottom_lip', 'left_eye', 'right_eye']:
                            # Create semi-transparent overlay
                            overlay = highlighted_face.copy()
                            cv2.fillPoly(overlay, [points_array], group_config['color'])
                            cv2.addWeighted(highlighted_face, 0.7, overlay, 0.3, 0, highlighted_face)
            
            feature_images[group_name] = highlighted_face
            
    except Exception as e:
        print(f"Error creating feature highlighted images with mesh: {str(e)}")
        # Return original face for all features if highlighting fails
        for group_name in feature_groups.keys():
            feature_images[group_name] = face_region.copy()
    
    return feature_images

def create_rating_highlighted_face_with_mesh(face_region, face_landmarks, face_left, face_top):
    """Create highlighted face image for rating display with mesh overlay"""
    try:
        highlighted_face = face_region.copy()
        
        # Convert landmarks to face-relative coordinates
        face_landmarks_adjusted = {}
        for feature_name, points in face_landmarks.items():
            face_points = []
            for point in points:
                face_x = point[0] - face_left
                face_y = point[1] - face_top
                if 0 <= face_x < face_region.shape[1] and 0 <= face_y < face_region.shape[0]:
                    face_points.append((face_x, face_y))
            if face_points:
                face_landmarks_adjusted[feature_name] = face_points
        
        # Convert to landmarks_points format for mesh drawing
        landmarks_points = []
        for feature_name, points in face_landmarks_adjusted.items():
            landmarks_points.extend(points)
        
        # Draw facial mesh first
        if landmarks_points:
            draw_facial_mesh(highlighted_face, landmarks_points)
        
        # Define colors for different features
        colors = {
            'chin': (255, 255, 255),      # Yellow
            'left_eyebrow': (255, 255, 255),  # Cyan
            'right_eyebrow': (255, 255, 255), # Cyan
            'nose_bridge': (255, 255, 255),     # Blue
            'nose_tip': (255, 255, 255),        # Red
            'left_eye': (255, 255, 255),        # Green
            'right_eye': (255, 255, 255),       # Green
            'top_lip': (255, 255, 255),       # Magenta
            'bottom_lip': (255, 255, 255)     # Purple
        }
        
        # Draw all facial features with enhanced highlighting
        for feature_name, points in face_landmarks_adjusted.items():
            color = colors.get(feature_name, (255, 255, 255))
            
            if len(points) > 2:
                points_array = np.array(points, dtype=np.int32)
                
                # Draw feature outline
                cv2.polylines(highlighted_face, [points_array], 
                            feature_name in ['top_lip', 'bottom_lip', 'left_eye', 'right_eye'], 
                            color, 2)
                
                # Add feature points
                for point in points:
                    cv2.circle(highlighted_face, tuple(map(int, point)), 2, color, -1)
        
        return highlighted_face
        
    except Exception as e:
        print(f"Error creating highlighted face with mesh: {str(e)}")
        return face_region.copy()

def download_image_from_url(image_url, timeout=30):
    """Download image from URL and return file path"""
    try:
        print(f"Downloading image from URL: {image_url}")
        
        # Validate URL
        if not image_url or not isinstance(image_url, str):
            raise ValueError("Invalid image URL provided")
        
        if not (image_url.startswith('http://') or image_url.startswith('https://')):
            raise ValueError("URL must start with http:// or https://")
        
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Download the image
        response = requests.get(image_url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp']):
            raise ValueError(f"Invalid content type: {content_type}. Expected image format.")
        
        # Check file size
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise ValueError(f"Image too large: {content_length} bytes. Maximum allowed: {MAX_FILE_SIZE} bytes.")
        
        # Generate unique filename
        file_extension = '.jpg'  # Default extension
        if 'image/png' in content_type:
            file_extension = '.png'
        elif 'image/gif' in content_type:
            file_extension = '.gif'
        elif 'image/webp' in content_type:
            file_extension = '.webp'
        elif 'image/bmp' in content_type:
            file_extension = '.bmp'
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"api_{timestamp}_{unique_id}{file_extension}"
        
        # Save the image
        filepath = safe_path_join(api_app.config['API_UPLOAD_FOLDER'], filename)
        
        total_size = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    total_size += len(chunk)
                    if total_size > MAX_FILE_SIZE:
                        f.close()
                        os.remove(filepath)
                        raise ValueError(f"Image too large: {total_size} bytes. Maximum allowed: {MAX_FILE_SIZE} bytes.")
                    f.write(chunk)
        
        print(f"Image downloaded successfully: {filepath} ({total_size} bytes)")
        
        # Verify the image can be opened
        try:
            with Image.open(filepath) as img:
                img.verify()
            print(f"Image verification successful: {img.format} {img.size}")
        except Exception as e:
            os.remove(filepath)
            raise ValueError(f"Downloaded file is not a valid image: {str(e)}")
        
        return filepath, filename
        
    except requests.exceptions.Timeout:
        raise ValueError(f"Timeout while downloading image from URL: {image_url}")
    except requests.exceptions.ConnectionError:
        raise ValueError(f"Connection error while downloading image from URL: {image_url}")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error while downloading image: {e.response.status_code} - {str(e)}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Request error while downloading image: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error downloading image from URL: {str(e)}")

def create_api_response(success, message, data=None, error_code=None):
    """Create standardized API response"""
    response = {
        'success': success,
        'message': message,
        'timestamp': datetime.now().isoformat(),
    }
    
    if data is not None:
        response['data'] = data
    
    if error_code is not None:
        response['error_code'] = error_code
    
    return response

def generate_public_urls(filename_base, face_data, tool_type):
    """Generate public URLs for processed images"""
    try:
        base_url = request.url_root.rstrip('/')
        public_urls = {}
        
        if tool_type == 'face_analysis':
            # Generate URLs for face analysis images
            for face in face_data:
                face_id = face['face_id']
                face_urls = {
                    'original': f"{base_url}/api/image/{filename_base}_face_{face_id}_original.png",
                    'features': {}
                }
                
                # Add feature-specific image URLs
                for feature_name in ['face_outline', 'eyes', 'eyebrows', 'nose', 'lips']:
                    face_urls['features'][feature_name] = f"{base_url}/api/image/{filename_base}_face_{face_id}_{feature_name}.png"
                
                public_urls[f'face_{face_id}'] = face_urls
                
        elif tool_type == 'face_rating':
            # Generate URLs for face rating images
            for face in face_data:
                face_id = face['face_id']
                public_urls[f'face_{face_id}'] = {
                    'highlighted': f"{base_url}/api/image/{filename_base}_face_{face_id}_highlighted.png"
                }
        
        return public_urls
        
    except Exception as e:
        print(f"Error generating public URLs: {str(e)}")
        return {}

# Import analysis functions from main app
def analyze_face_shape(landmarks):
    """Analyze face shape based on facial landmarks"""
    try:
        if len(landmarks) < 68:
            return create_fallback_analysis("Face Shape")
        
        # Key points for face shape analysis with safe access
        chin = safe_get_landmark(landmarks, 8)
        left_cheek = safe_get_landmark(landmarks, 1)
        right_cheek = safe_get_landmark(landmarks, 15)
        forehead_left = safe_get_landmark(landmarks, 19)
        forehead_right = safe_get_landmark(landmarks, 24)
        jaw_left = safe_get_landmark(landmarks, 5)
        jaw_right = safe_get_landmark(landmarks, 11)
        
        # Calculate face measurements
        face_width = calculate_distance(left_cheek, right_cheek)
        face_length = calculate_distance(forehead_left, chin)
        jaw_width = calculate_distance(jaw_left, jaw_right)
        forehead_width = calculate_distance(forehead_left, forehead_right)
        
        # Prevent division by zero
        width_to_length_ratio = face_width / face_length if face_length > 0 else 0
        jaw_to_forehead_ratio = jaw_width / forehead_width if forehead_width > 0 else 0
        
        # Determine face shape
        if width_to_length_ratio > 0.9:
            if jaw_to_forehead_ratio > 0.9:
                face_shape = "Square"
                description = "You have a square face with strong, defined angles."
            else:
                face_shape = "Round"
                description = "You have a round face with soft, curved features."
        elif width_to_length_ratio < 0.7:
            face_shape = "Oval"
            description = "You have an oval face with balanced proportions."
        else:
            if jaw_to_forehead_ratio < 0.8:
                face_shape = "Heart"
                description = "You have a heart-shaped face with a wider forehead."
            else:
                face_shape = "Rectangle"
                description = "You have a rectangular face with elongated features."
        
        # Face feature analysis
        chin_analysis = "Square" if jaw_to_forehead_ratio > 0.9 else "Pointed" if jaw_to_forehead_ratio < 0.7 else "Rounded"
        cheekbone_analysis = "High" if width_to_length_ratio > 0.8 else "Flat"
        temple_analysis = "Wide" if forehead_width > face_width * 0.8 else "Normal"
        
        return {
            'shape': face_shape,
            'description': description,
            'measurements': {
                'width': round(face_width, 1),
                'length': round(face_length, 1),
                'jaw_width': round(jaw_width, 1),
                'forehead_width': round(forehead_width, 1),
                'width_to_length_ratio': round(width_to_length_ratio, 2)
            },
            'characteristics': {
                'chin': chin_analysis,
                'cheekbone': cheekbone_analysis,
                'temple': temple_analysis,
                'apple_cheeks': "Prominent" if cheekbone_analysis == "High" else "Flat"
            },
            'recommendations': get_face_shape_recommendations(face_shape)
        }
    except Exception as e:
        print(f"Error in face shape analysis: {str(e)}")
        return create_fallback_analysis("Face Shape")

def create_fallback_analysis(feature_name):
    """Create fallback analysis when detailed analysis fails"""
    return {
        'shape': 'Unknown',
        'description': f'{feature_name} analysis not available.',
        'measurements': {
            'width': 0,
            'length': 0,
            'jaw_width': 0,
            'forehead_width': 0,
            'width_to_length_ratio': 0
        },
        'characteristics': {
            'chin': 'Unknown',
            'cheekbone': 'Unknown',
            'temple': 'Unknown',
            'apple_cheeks': 'Unknown'
        },
        'recommendations': ['Analysis not available']
    }

def get_face_shape_recommendations(face_shape):
    """Get styling recommendations based on face shape"""
    recommendations = {
        'Square': ['Soften angles with rounded frames', 'Add width to upper face', 'Avoid sharp, angular styles'],
        'Round': ['Add angles and definition', 'Create vertical lines', 'Avoid round, circular shapes'],
        'Oval': ['Most styles work well', 'Maintain natural balance', 'Experiment with different looks'],
        'Heart': ['Balance wider forehead', 'Add width to lower face', 'Soften the chin area'],
        'Rectangle': ['Add width to face', 'Break up length', 'Create horizontal emphasis']
    }
    return recommendations.get(face_shape, ['Consult with a styling professional'])

def analyze_eyes(landmarks):
    """Analyze eye characteristics"""
    try:
        if len(landmarks) < 68:
            return create_eye_fallback()
        
        # Left eye landmarks (indices 36-41) and right eye (42-47)
        left_eye_points = landmarks[36:42] if len(landmarks) > 41 else []
        right_eye_points = landmarks[42:48] if len(landmarks) > 47 else []
        
        if len(left_eye_points) < 6 or len(right_eye_points) < 6:
            return create_eye_fallback()
        
        # Calculate eye measurements
        left_eye_width = calculate_distance(left_eye_points[0], left_eye_points[3])
        left_eye_height = calculate_distance(left_eye_points[1], left_eye_points[5])
        right_eye_width = calculate_distance(right_eye_points[0], right_eye_points[3])
        right_eye_height = calculate_distance(right_eye_points[1], right_eye_points[5])
        
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        # Eye spacing
        eye_distance = calculate_distance(left_eye_points[3], right_eye_points[0])
        
        # Determine eye characteristics
        eye_aspect_ratio = avg_eye_width / avg_eye_height if avg_eye_height > 0 else 0
        
        if eye_aspect_ratio > 3:
            eye_shape = "Almond"
        elif eye_aspect_ratio > 2.5:
            eye_shape = "Oval"
        else:
            eye_shape = "Round"
        
        if avg_eye_width < 25:
            eye_size = "Small"
        elif avg_eye_width > 35:
            eye_size = "Large"
        else:
            eye_size = "Medium"
        
        if eye_distance < avg_eye_width:
            eye_spacing = "Close"
        elif eye_distance > avg_eye_width * 1.5:
            eye_spacing = "Wide"
        else:
            eye_spacing = "Average"

        return {
            'characteristics': {
                'size': eye_size,
                'shape': eye_shape,
                'spacing': eye_spacing
            },
            'measurements': {
                'distance': round(eye_distance, 1),
                'avg_width': round(avg_eye_width, 1),
                'avg_height': round(avg_eye_height, 1),
                'left_width': round(left_eye_width, 1),
                'right_width': round(right_eye_width, 1)
            },
            'description': f"You have {eye_size.lower()}, {eye_shape.lower()}-shaped eyes with {eye_spacing.lower()} spacing."
        }
    except Exception as e:
        print(f"Error in eye analysis: {str(e)}")
        return create_eye_fallback()

def create_eye_fallback():
    """Create fallback eye analysis"""
    return {
        'characteristics': {
            'size': 'Unknown',
            'shape': 'Unknown',
            'spacing': 'Unknown'
        },
        'measurements': {
            'distance': 0,
            'avg_width': 0,
            'avg_height': 0,
            'left_width': 0,
            'right_width': 0
        },
        'description': "Eye analysis not available."
    }

def analyze_eyebrows(landmarks):
    """Analyze eyebrow characteristics"""
    try:
        if len(landmarks) < 68:
            return create_eyebrow_fallback()
        
        # Eyebrow landmarks
        left_eyebrow = landmarks[17:22] if len(landmarks) > 21 else []
        right_eyebrow = landmarks[22:27] if len(landmarks) > 26 else []
        
        if len(left_eyebrow) < 5 or len(right_eyebrow) < 5:
            return create_eyebrow_fallback()
        
        # Calculate eyebrow measurements
        left_brow_length = calculate_distance(left_eyebrow[0], left_eyebrow[4])
        right_brow_length = calculate_distance(right_eyebrow[0], right_eyebrow[4])
        avg_brow_length = (left_brow_length + right_brow_length) / 2
        
        # Eyebrow height (thickness approximation)
        left_brow_height = abs(left_eyebrow[2][1] - landmarks[19][1]) if len(landmarks) > 19 else 0
        right_brow_height = abs(right_eyebrow[2][1] - landmarks[24][1]) if len(landmarks) > 24 else 0
        avg_brow_height = (left_brow_height + right_brow_height) / 2
        
        # Eyebrow spacing
        brow_spacing = calculate_distance(left_eyebrow[4], right_eyebrow[0])
        
        # Determine characteristics
        if avg_brow_height < 8:
            thickness = "Thin"
        elif avg_brow_height > 15:
            thickness = "Thick"
        else:
            thickness = "Medium"
        
        # Arch analysis (simplified)
        left_arch_height = left_eyebrow[2][1] - min(p[1] for p in left_eyebrow)
        right_arch_height = right_eyebrow[2][1] - min(p[1] for p in right_eyebrow)
        avg_arch = (left_arch_height + right_arch_height) / 2
        
        if avg_arch > 5:
            arch = "High"
        elif avg_arch < 2:
            arch = "Straight"
        else:
            arch = "Medium"
        
        if brow_spacing > avg_brow_length * 0.3:
            spacing = "Wide spacing"
        else:
            spacing = "Close spacing"
        
        return {
            'characteristics': {
                'thickness': thickness,
                'arch': arch,
                'spacing': spacing
            },
            'measurements': {
                'length': round(avg_brow_length, 1),
                'height': round(avg_brow_height, 1),
                'spacing': round(brow_spacing, 1)
            },
            'description': f"You have {thickness.lower()} eyebrows with {arch.lower()} arch and {spacing}."
        }
    except Exception as e:
        print(f"Error in eyebrow analysis: {str(e)}")
        return create_eyebrow_fallback()

def create_eyebrow_fallback():
    """Create fallback eyebrow analysis"""
    return {
        'characteristics': {
            'thickness': 'Unknown',
            'arch': 'Unknown',
            'spacing': 'Unknown'
        },
        'measurements': {
            'length': 0,
            'height': 0,
            'spacing': 0
        },
        'description': "Eyebrow analysis not available."
    }

def analyze_nose(landmarks):
    """Analyze nose characteristics"""
    try:
        if len(landmarks) < 68:
            return create_nose_fallback()
        
        # Nose landmarks
        nose_bridge = landmarks[27:31] if len(landmarks) > 30 else []
        nose_tip = landmarks[30:36] if len(landmarks) > 35 else []
        
        if len(nose_bridge) < 4 or len(nose_tip) < 6:
            return create_nose_fallback()
        
        # Calculate nose measurements
        nose_width = calculate_distance(nose_tip[1], nose_tip[5])  # Nostril width
        nose_length = calculate_distance(nose_bridge[0], nose_tip[3])  # Bridge to tip
        bridge_width = calculate_distance(landmarks[31], landmarks[35]) if len(landmarks) > 35 else 0
        
        # Nose ratios
        width_ratio = nose_width / nose_length if nose_length > 0 else 0
        
        # Determine characteristics
        if nose_width < 30:
            width_desc = "Narrow"
        elif nose_width > 45:
            width_desc = "Wide"
        else:
            width_desc = "Medium"
        
        if nose_length < 40:
            length_desc = "Short"
        elif nose_length > 60:
            length_desc = "Long"
        else:
            length_desc = "Medium"
        
        # Bridge analysis
        bridge_height = abs(landmarks[27][1] - landmarks[30][1]) if len(landmarks) > 30 else 0
        if bridge_height < 15:
            bridge_desc = "Low bridge"
        elif bridge_height > 25:
            bridge_desc = "High bridge"
        else:
            bridge_desc = "Medium bridge"
        
        # Shape analysis (simplified)
        tip_width = calculate_distance(landmarks[31], landmarks[35]) if len(landmarks) > 35 else 0
        if tip_width / nose_width > 0.7 if nose_width > 0 else False:
            shape_desc = "Straight"
        else:
            shape_desc = "Curved"
        
        return {
            'characteristics': {
                'width': width_desc,
                'length': length_desc,
                'shape': shape_desc,
                'bridge': bridge_desc
            },
            'measurements': {
                'width': round(nose_width, 1),
                'height': round(nose_length, 1),
                'bridge_width': round(bridge_width, 1),
                'width_ratio': round(width_ratio, 2)
            },
            'description': f"You have a {width_desc.lower()}, {length_desc.lower()} nose with a {shape_desc.lower()} shape and {bridge_desc}."
        }
    except Exception as e:
        print(f"Error in nose analysis: {str(e)}")
        return create_nose_fallback()

def create_nose_fallback():
    """Create fallback nose analysis"""
    return {
        'characteristics': {
            'width': 'Unknown',
            'length': 'Unknown',
            'shape': 'Unknown',
            'bridge': 'Unknown'
        },
        'measurements': {
            'width': 0,
            'height': 0,
            'bridge_width': 0,
            'width_ratio': 0
        },
        'description': "Nose analysis not available."
    }

def analyze_lips(landmarks):
    """Analyze lip characteristics"""
    try:
        if len(landmarks) < 68:
            return create_lips_fallback()
        
        # Lip landmarks
        outer_lip = landmarks[48:60] if len(landmarks) > 59 else []
        inner_lip = landmarks[60:68] if len(landmarks) > 67 else []
        
        if len(outer_lip) < 12 or len(inner_lip) < 8:
            return create_lips_fallback()
        
        # Calculate lip measurements
        lip_width = calculate_distance(outer_lip[0], outer_lip[6])
        upper_lip_height = calculate_distance(outer_lip[3], inner_lip[1])
        lower_lip_height = calculate_distance(outer_lip[9], inner_lip[4])
        total_lip_height = calculate_distance(outer_lip[3], outer_lip[9])
        
        # Lip ratios
        width_ratio = lip_width / total_lip_height if total_lip_height > 0 else 0
        upper_lower_ratio = upper_lip_height / lower_lip_height if lower_lip_height > 0 else 0
        
        # Determine characteristics
        if lip_width < 45:
            width_desc = "Narrow"
        elif lip_width > 65:
            width_desc = "Wide"
        else:
            width_desc = "Medium"
        
        if total_lip_height < 20:
            thickness_desc = "Thin"
        elif total_lip_height > 35:
            thickness_desc = "Full"
        else:
            thickness_desc = "Medium"
        
        # Shape analysis
        if upper_lower_ratio > 1.2:
            shape_desc = "Top-heavy"
        elif upper_lower_ratio < 0.8:
            shape_desc = "Bottom-heavy"
        else:
            shape_desc = "Balanced"
        
        # Cupid's bow analysis (simplified)
        cupid_bow_depth = abs(outer_lip[2][1] - outer_lip[3][1]) if len(outer_lip) > 3 else 0
        if cupid_bow_depth > 3:
            cupid_bow = "Pronounced"
        else:
            cupid_bow = "Subtle"
        
        return {
            'characteristics': {
                'width': width_desc,
                'thickness': thickness_desc,
                'shape': shape_desc,
                'cupid_bow': cupid_bow
            },
            'measurements': {
                'width': round(lip_width, 1),
                'height': round(total_lip_height, 1),
                'upper_height': round(upper_lip_height, 1),
                'lower_height': round(lower_lip_height, 1),
                'width_ratio': round(width_ratio, 2)
            },
            'description': f"You have {width_desc.lower()}, {thickness_desc.lower()} lips with a {shape_desc.lower()} shape."
        }
    except Exception as e:
        print(f"Error in lips analysis: {str(e)}")
        return create_lips_fallback()

def create_lips_fallback():
    """Create fallback lips analysis"""
    return {
        'characteristics': {
            'width': 'Unknown',
            'thickness': 'Unknown',
            'shape': 'Unknown',
            'cupid_bow': 'Unknown'
        },
        'measurements': {
            'width': 0,
            'height': 0,
            'upper_height': 0,
            'lower_height': 0,
            'width_ratio': 0
        },
        'description': "Lip analysis not available."
    }

def calculate_overall_attractiveness_score(face_shape, eyes, nose, lips):
    """Calculate an overall attractiveness score based on facial features"""
    try:
        score = 0
        max_score = 100
        
        # Face shape scoring (20 points)
        if face_shape.get('shape') in ['Oval', 'Heart']:
            score += 20
        elif face_shape.get('shape') in ['Square', 'Rectangle']:
            score += 15
        else:
            score += 10
        
        # Eye scoring (25 points)
        eye_chars = eyes.get('characteristics', {})
        if eye_chars.get('spacing') == 'Average':
            score += 10
        if eye_chars.get('size') in ['Medium', 'Large']:
            score += 10
        if eye_chars.get('shape') in ['Almond', 'Oval']:
            score += 5
        
        # Nose scoring (25 points)
        nose_chars = nose.get('characteristics', {})
        if nose_chars.get('width') == 'Medium':
            score += 10
        if nose_chars.get('length') == 'Medium':
            score += 10
        nose_measurements = nose.get('measurements', {})
        width_ratio = nose_measurements.get('width_ratio', 0)
        if 0.1 < width_ratio < 0.2:
            score += 5
        
        # Lip scoring (30 points)
        lip_chars = lips.get('characteristics', {})
        if lip_chars.get('thickness') in ['Medium', 'Full']:
            score += 15
        if lip_chars.get('shape') == 'Balanced':
            score += 10
        if lip_chars.get('width') == 'Medium':
            score += 5
        
        return min(score, max_score)
    except Exception as e:
        print(f"Error calculating score: {str(e)}")
        return 50

def comprehensive_face_analysis(image_path):
    """Perform comprehensive face analysis with mesh highlighting"""
    try:
        print(f"Starting API analysis of: {image_path}")
        
        # Verify file exists
        if not os.path.exists(image_path):
            print(f"Error: File does not exist: {image_path}")
            return None, 0, [], []
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None, 0, [], []
        
        print(f"Image loaded successfully: {image.shape}")
        
        # Convert BGR to RGB for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find face locations and landmarks
        print("Detecting faces...")
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        print(f"Found {len(face_locations)} faces")
        
        if len(face_locations) == 0:
            print("No faces detected")
            return image, 0, [], []
        
        face_landmarks_list = face_recognition.face_landmarks(rgb_image, face_locations)
        print(f"Generated {len(face_landmarks_list)} landmark sets")
        
        processed_image = image.copy()
        analysis_results = []
        extracted_faces = []
        
        for face_idx, (face_landmarks, face_location) in enumerate(zip(face_landmarks_list, face_locations)):
            print(f"Processing face {face_idx + 1}")
            
            top, right, bottom, left = face_location
            
            # Add padding
            padding = 30
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(image.shape[1], right + padding)
            bottom = min(image.shape[0], bottom + padding)
            
            # Draw face bounding box
            cv2.rectangle(processed_image, (left, top), (right, bottom), (255, 255, 255), 3)
            cv2.putText(processed_image, f'Person {face_idx + 1}', (left, top - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Extract face region
            face_region = image[top:bottom, left:right].copy()
            
            # Convert landmarks to list of tuples for analysis
            landmarks_points = []
            for feature_name, points in face_landmarks.items():
                landmarks_points.extend(points)
            
            print(f"Landmarks extracted: {len(landmarks_points)} points")
            
            # Perform detailed analysis
            try:
                face_shape_analysis = analyze_face_shape(landmarks_points)
                eye_analysis = analyze_eyes(landmarks_points)
                eyebrow_analysis = analyze_eyebrows(landmarks_points)
                nose_analysis = analyze_nose(landmarks_points)
                lip_analysis = analyze_lips(landmarks_points)
                
                print(f"Analysis completed for face {face_idx + 1}")
                
                # Create separate highlighted images for each feature with mesh overlay
                feature_images = create_feature_highlighted_images_with_mesh(face_region, face_landmarks, left, top)
                
                # Draw facial mesh on main processed image for overview
                draw_facial_mesh_on_main_image(processed_image, landmarks_points)
                
                # Compile comprehensive analysis
                comprehensive_analysis = {
                    'face_id': face_idx + 1,
                    'bbox': (left, top, right, bottom),
                    'face_shape': face_shape_analysis,
                    'eyes': eye_analysis,
                    'eyebrows': eyebrow_analysis,
                    'nose': nose_analysis,
                    'lips': lip_analysis,
                    'overall_score': calculate_overall_attractiveness_score(
                        face_shape_analysis, eye_analysis, nose_analysis, lip_analysis
                    )
                }
                
                analysis_results.append(comprehensive_analysis)
                print(f"Analysis result added for face {face_idx + 1}")
                
            except Exception as e:
                print(f"Error in detailed analysis for face {face_idx + 1}: {str(e)}")
                # Fallback to basic analysis
                basic_analysis = {
                    'face_id': face_idx + 1,
                    'bbox': (left, top, right, bottom),
                    'face_shape': create_fallback_analysis("Face Shape"),
                    'eyes': create_eye_fallback(),
                    'eyebrows': create_eyebrow_fallback(),
                    'nose': create_nose_fallback(),
                    'lips': create_lips_fallback(),
                    'overall_score': 50,
                    'error': 'Detailed analysis failed, showing basic detection only'
                }
                analysis_results.append(basic_analysis)
            
            # Store extracted faces with individual feature highlights
            extracted_faces.append({
                'original': face_region,
                'feature_images': feature_images,
                'face_id': face_idx + 1
            })
        
        print(f"Analysis complete. Found {len(analysis_results)} faces")
        return processed_image, len(analysis_results), analysis_results, extracted_faces
    
    except Exception as e:
        print(f"Error in comprehensive face analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, 0, [], []

def draw_facial_mesh_on_main_image(image, landmarks_points):
    """Draw facial mesh on the main processed image"""
    try:
        # Draw the facial mesh directly on the main image
        draw_facial_mesh(image, landmarks_points)
    except Exception as e:
        print(f"Error drawing mesh on main image: {str(e)}")

def perform_face_rating_analysis(image_path, filename_base):
    """Perform comprehensive face rating analysis for multiple faces with mesh highlighting"""
    try:
        print(f"Starting multi-face rating analysis of: {image_path}")
        
        # Verify file exists
        if not os.path.exists(image_path):
            print(f"Error: File does not exist: {image_path}")
            return {'success': False, 'message': 'File not found'}
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return {'success': False, 'message': 'Could not load image'}
        
        print(f"Image loaded successfully: {image.shape}")
        
        # Convert BGR to RGB for face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find face locations and landmarks
        print("Detecting faces for rating...")
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        print(f"Found {len(face_locations)} faces")
        
        if len(face_locations) == 0:
            print("No faces detected")
            return {'success': False, 'message': 'No faces detected in the image'}
        
        face_landmarks_list = face_recognition.face_landmarks(rgb_image, face_locations)
        print(f"Generated {len(face_landmarks_list)} landmark sets")
        
        # Analyze all faces
        individual_ratings = []
        highlighted_images = []
        
        for face_idx, (face_landmarks, face_location) in enumerate(zip(face_landmarks_list, face_locations)):
            print(f"Rating face {face_idx + 1}")
            
            top, right, bottom, left = face_location
            
            # Add padding
            padding = 30
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(image.shape[1], right + padding)
            bottom = min(image.shape[0], bottom + padding)
            
            # Extract face region
            face_region = image[top:bottom, left:right].copy()
            
            # Convert landmarks to list of tuples for analysis
            landmarks_points = []
            for feature_name, points in face_landmarks.items():
                landmarks_points.extend(points)
            
            print(f"Landmarks extracted: {len(landmarks_points)} points for face {face_idx + 1}")
            
            # Perform detailed rating analysis for this face
            face_rating_analysis = analyze_face_for_rating(landmarks_points, face_region, image.shape)
            face_rating_analysis['face_id'] = face_idx + 1
            
            # Create highlighted face image for this face with mesh
            highlighted_face = create_rating_highlighted_face_with_mesh(face_region, face_landmarks, left, top)
            
            # Save highlighted face image
            highlighted_filename = f"{filename_base}_face_{face_idx + 1}_highlighted.png"
            highlighted_path = safe_path_join(API_PROCESSED_FOLDER, highlighted_filename)
            os.makedirs(os.path.dirname(highlighted_path), exist_ok=True)
            cv2.imwrite(highlighted_path, highlighted_face)
            
            face_rating_analysis['highlighted_image'] = highlighted_filename
            individual_ratings.append(face_rating_analysis)
            
            print(f"Face {face_idx + 1} rating: {face_rating_analysis['overall_rating']}/10")
        
        # Calculate summary statistics
        total_faces = len(individual_ratings)
        ratings = [face['overall_rating'] for face in individual_ratings]
        percentages = [face['percentage'] for face in individual_ratings]
        
        average_rating = round(sum(ratings) / len(ratings), 1) if ratings else 0
        average_percentage = round(sum(percentages) / len(percentages)) if percentages else 0
        highest_rated = max(individual_ratings, key=lambda x: x['overall_rating']) if individual_ratings else None
        
        # Compile comprehensive rating data
        comprehensive_rating_data = {
            'face_count': total_faces,
            'average_rating': average_rating,
            'average_percentage': average_percentage,
            'highest_rated': highest_rated,
            'individual_ratings': individual_ratings
        }
        
        print(f"Multi-face rating analysis completed")
        print(f"Summary: {total_faces} faces, average rating: {average_rating}/10")
        
        return {
            'success': True,
            'data': comprehensive_rating_data
        }
        
    except Exception as e:
        print(f"Error in multi-face rating analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': f'Analysis failed: {str(e)}'}

def analyze_face_for_rating(landmarks, face_region, image_shape):
    """Analyze face for attractiveness rating"""
    try:
        if len(landmarks) < 68:
            return create_default_rating()
        
        # Calculate facial measurements
        measurements = calculate_facial_measurements(landmarks)
        
        # Analyze facial proportions
        proportions = analyze_facial_proportions(measurements)
        
        # Calculate symmetry
        symmetry_score = calculate_face_symmetry(landmarks)
        
        # Generate feature analysis
        feature_analysis = generate_feature_analysis(measurements, proportions)
        
        # Calculate overall rating
        overall_rating = calculate_overall_rating(proportions, symmetry_score, measurements)
        
        # Generate rating message
        rating_message = generate_rating_message(overall_rating)
        
        # Calculate percentage
        percentage = int((overall_rating / 10.0) * 100)
        
        return {
            'overall_rating': round(overall_rating, 1),
            'percentage': percentage,
            'rating_message': rating_message,
            'feature_analysis': feature_analysis,
            'symmetry_score': symmetry_score,
            'measurements': measurements,
            'proportions': proportions
        }
        
    except Exception as e:
        print(f"Error in rating analysis: {str(e)}")
        return create_default_rating()

def calculate_facial_measurements(landmarks):
    """Calculate key facial measurements"""
    try:
        measurements = {}
        
        # Face width and height
        left_face = safe_get_landmark(landmarks, 1)
        right_face = safe_get_landmark(landmarks, 15)
        top_face = safe_get_landmark(landmarks, 19)
        bottom_face = safe_get_landmark(landmarks, 8)
        
        measurements['face_width'] = calculate_distance(left_face, right_face)
        measurements['face_height'] = calculate_distance(top_face, bottom_face)
        
        # Forehead measurements
        forehead_left = safe_get_landmark(landmarks, 17)
        forehead_right = safe_get_landmark(landmarks, 26)
        measurements['forehead_width'] = calculate_distance(forehead_left, forehead_right)
        
        # Eye measurements
        left_eye_inner = safe_get_landmark(landmarks, 39)
        right_eye_inner = safe_get_landmark(landmarks, 42)
        measurements['interocular_distance'] = calculate_distance(left_eye_inner, right_eye_inner)
        
        left_eye_outer = safe_get_landmark(landmarks, 36)
        right_eye_outer = safe_get_landmark(landmarks, 45)
        measurements['eye_span'] = calculate_distance(left_eye_outer, right_eye_outer)
        
        # Nose measurements
        nose_top = safe_get_landmark(landmarks, 27)
        nose_bottom = safe_get_landmark(landmarks, 33)
        nose_left = safe_get_landmark(landmarks, 31)
        nose_right = safe_get_landmark(landmarks, 35)
        
        measurements['nose_length'] = calculate_distance(nose_top, nose_bottom)
        measurements['nose_width'] = calculate_distance(nose_left, nose_right)
        
        # Mouth measurements
        mouth_left = safe_get_landmark(landmarks, 48)
        mouth_right = safe_get_landmark(landmarks, 54)
        measurements['mouth_width'] = calculate_distance(mouth_left, mouth_right)
        
        # Chin measurements
        chin_point = safe_get_landmark(landmarks, 8)
        jaw_left = safe_get_landmark(landmarks, 5)
        jaw_right = safe_get_landmark(landmarks, 11)
        measurements['jaw_width'] = calculate_distance(jaw_left, jaw_right)
        
        return measurements
        
    except Exception as e:
        print(f"Error calculating measurements: {str(e)}")
        return {}

def analyze_facial_proportions(measurements):
    """Analyze facial proportions for rating"""
    try:
        proportions = {}
        
        if measurements.get('face_width', 0) > 0 and measurements.get('face_height', 0) > 0:
            proportions['face_ratio'] = measurements['face_width'] / measurements['face_height']
        else:
            proportions['face_ratio'] = 0
        
        if measurements.get('face_width', 0) > 0 and measurements.get('forehead_width', 0) > 0:
            proportions['forehead_ratio'] = measurements['forehead_width'] / measurements['face_width']
        else:
            proportions['forehead_ratio'] = 0
        
        if measurements.get('face_width', 0) > 0 and measurements.get('interocular_distance', 0) > 0:
            proportions['eye_spacing_ratio'] = measurements['interocular_distance'] / measurements['face_width']
        else:
            proportions['eye_spacing_ratio'] = 0
        
        if measurements.get('face_height', 0) > 0 and measurements.get('nose_length', 0) > 0:
            proportions['nose_length_ratio'] = measurements['nose_length'] / measurements['face_height']
        else:
            proportions['nose_length_ratio'] = 0
        
        if measurements.get('face_width', 0) > 0 and measurements.get('nose_width', 0) > 0:
            proportions['nose_width_ratio'] = measurements['nose_width'] / measurements['face_width']
        else:
            proportions['nose_width_ratio'] = 0
        
        if measurements.get('face_width', 0) > 0 and measurements.get('mouth_width', 0) > 0:
            proportions['mouth_width_ratio'] = measurements['mouth_width'] / measurements['face_width']
        else:
            proportions['mouth_width_ratio'] = 0
        
        if measurements.get('face_width', 0) > 0 and measurements.get('jaw_width', 0) > 0:
            proportions['jaw_width_ratio'] = measurements['jaw_width'] / measurements['face_width']
        else:
            proportions['jaw_width_ratio'] = 0
        
        return proportions
        
    except Exception as e:
        print(f"Error analyzing proportions: {str(e)}")
        return {}

def calculate_face_symmetry(landmarks):
    """Calculate facial symmetry score"""
    try:
        if len(landmarks) < 68:
            return 50
        
        # Get center line of face
        nose_tip = safe_get_landmark(landmarks, 33)
        chin = safe_get_landmark(landmarks, 8)
        center_x = (nose_tip[0] + chin[0]) / 2
        
        # Calculate symmetry for key features
        symmetry_scores = []
        
        # Eye symmetry
        left_eye_center = safe_get_landmark(landmarks, 37)
        right_eye_center = safe_get_landmark(landmarks, 44)
        left_distance = abs(left_eye_center[0] - center_x)
        right_distance = abs(right_eye_center[0] - center_x)
        eye_symmetry = 100 - min(abs(left_distance - right_distance) * 2, 100)
        symmetry_scores.append(eye_symmetry)
        
        # Eyebrow symmetry
        left_brow = safe_get_landmark(landmarks, 19)
        right_brow = safe_get_landmark(landmarks, 24)
        left_brow_distance = abs(left_brow[0] - center_x)
        right_brow_distance = abs(right_brow[0] - center_x)
        brow_symmetry = 100 - min(abs(left_brow_distance - right_brow_distance) * 2, 100)
        symmetry_scores.append(brow_symmetry)
        
        # Mouth symmetry
        left_mouth = safe_get_landmark(landmarks, 48)
        right_mouth = safe_get_landmark(landmarks, 54)
        left_mouth_distance = abs(left_mouth[0] - center_x)
        right_mouth_distance = abs(right_mouth[0] - center_x)
        mouth_symmetry = 100 - min(abs(left_mouth_distance - right_mouth_distance) * 2, 100)
        symmetry_scores.append(mouth_symmetry)
        
        # Average symmetry
        overall_symmetry = sum(symmetry_scores) / len(symmetry_scores)
        return max(0, min(100, overall_symmetry))
        
    except Exception as e:
        print(f"Error calculating symmetry: {str(e)}")
        return 50

def generate_feature_analysis(measurements, proportions):
    """Generate detailed feature analysis"""
    try:
        analysis = []
        
        # Face width analysis
        face_ratio = proportions.get('face_ratio', 0)
        if face_ratio > 0.85:
            analysis.append("Face too wide")
        elif face_ratio < 0.75:
            analysis.append("Face too narrow")
        else:
            analysis.append("Good face width")
        
        # Forehead analysis
        forehead_ratio = proportions.get('forehead_ratio', 0)
        if forehead_ratio < 0.6:
            analysis.append("Small forehead")
        elif forehead_ratio > 0.8:
            analysis.append("Large forehead")
        else:
            analysis.append("Good forehead size")
        
        # Eye spacing analysis
        eye_spacing_ratio = proportions.get('eye_spacing_ratio', 0)
        if eye_spacing_ratio < 0.25:
            analysis.append("Narrow interocular distance")
        elif eye_spacing_ratio > 0.35:
            analysis.append("Wide interocular distance")
        else:
            analysis.append("Good eye spacing")
        
        # Nose analysis
        nose_width_ratio = proportions.get('nose_width_ratio', 0)
        nose_length_ratio = proportions.get('nose_length_ratio', 0)
        
        if 0.15 <= nose_width_ratio <= 0.25:
            analysis.append("Good nose for face")
        elif nose_width_ratio > 0.25:
            analysis.append("Nose too wide for face")
        else:
            analysis.append("Nose too narrow for face")
        
        if nose_length_ratio > 0.35:
            analysis.append("Nose too long for face")
        elif nose_length_ratio < 0.25:
            analysis.append("Nose too short for face")
        else:
            analysis.append("Good nose length")
        
        # Mouth analysis
        mouth_width_ratio = proportions.get('mouth_width_ratio', 0)
        if mouth_width_ratio > 0.55:
            analysis.append("Mouth too wide")
        elif mouth_width_ratio < 0.45:
            analysis.append("Mouth too narrow")
        else:
            analysis.append("Good mouth width")
        
        # Jaw analysis
        jaw_width_ratio = proportions.get('jaw_width_ratio', 0)
        if jaw_width_ratio > 0.85:
            analysis.append("Chin too big")
        elif jaw_width_ratio < 0.65:
            analysis.append("Chin too small")
        else:
            analysis.append("Good chin size")
        
        return analysis
        
    except Exception as e:
        print(f"Error generating feature analysis: {str(e)}")
        return ["Analysis not available"]

def calculate_overall_rating(proportions, symmetry_score, measurements):
    """Calculate overall attractiveness rating"""
    try:
        score = 5.0  # Base score
        
        # Face ratio scoring
        face_ratio = proportions.get('face_ratio', 0)
        if 0.75 <= face_ratio <= 0.85:
            score += 1.0
        elif 0.7 <= face_ratio <= 0.9:
            score += 0.5
        else:
            score -= 0.5
        
        # Forehead ratio scoring
        forehead_ratio = proportions.get('forehead_ratio', 0)
        if 0.6 <= forehead_ratio <= 0.8:
            score += 0.8
        else:
            score -= 0.3
        
        # Eye spacing scoring
        eye_spacing_ratio = proportions.get('eye_spacing_ratio', 0)
        if 0.25 <= eye_spacing_ratio <= 0.35:
            score += 0.7
        else:
            score -= 0.4
        
        # Nose scoring
        nose_width_ratio = proportions.get('nose_width_ratio', 0)
        if 0.15 <= nose_width_ratio <= 0.25:
            score += 0.8
        else:
            score -= 0.4
        
        # Mouth scoring
        mouth_width_ratio = proportions.get('mouth_width_ratio', 0)
        if 0.45 <= mouth_width_ratio <= 0.55:
            score += 0.6
        else:
            score -= 0.3
        
        # Symmetry scoring
        if symmetry_score >= 80:
            score += 1.2
        elif symmetry_score >= 60:
            score += 0.6
        else:
            score -= 0.8
        
        # Ensure score is within bounds
        score = max(1.0, min(10.0, score))
        
        return score
        
    except Exception as e:
        print(f"Error calculating overall rating: {str(e)}")
        return 5.0

def generate_rating_message(rating):
    """Generate rating message based on score"""
    if rating >= 8.5:
        return "You are exceptionally attractive!"
    elif rating >= 7.5:
        return "You are very attractive!"
    elif rating >= 6.5:
        return "You are good looking!"
    elif rating >= 5.5:
        return "You have attractive features!"
    elif rating >= 4.5:
        return "You have decent features!"
    else:
        return "You have unique features!"

def create_default_rating():
    """Create default rating when analysis fails"""
    return {
        'overall_rating': 5.0,
        'percentage': 50,
        'rating_message': 'Analysis not available',
        'feature_analysis': ['Analysis not available'],
        'symmetry_score': 50,
        'measurements': {},
        'proportions': {}
    }

@api_app.route('/api/health', methods=['GET'])
def api_health():
    """API health check endpoint"""
    return jsonify(create_api_response(
        success=True,
        message="Face Analysis API is running",
        data={
            'version': '1.0.0',
            'available_tools': ['face_analysis', 'face_rating'],
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'max_file_size_mb': MAX_FILE_SIZE // (1024 * 1024)
        }
    ))

def save_api_extracted_faces(extracted_faces, filename_base):
    """Save extracted faces for API with proper paths and mesh highlighting"""
    saved_faces = []
    
    try:
        # Ensure API directories exist
        ensure_api_directories()
        
        for face_data in extracted_faces:
            face_id = face_data['face_id']
            
            # Save original face to API faces folder
            original_filename = f"{filename_base}_face_{face_id}_original.png"
            original_path = safe_path_join(API_FACES_FOLDER, original_filename)
            success_original = cv2.imwrite(original_path, face_data['original'])
            
            # Save individual feature highlighted images to API processed folder
            feature_files = {}
            feature_images_b64 = {}
            
            for feature_name, feature_image in face_data['feature_images'].items():
                feature_filename = f"{filename_base}_face_{face_id}_{feature_name}.png"
                
                # Save to API processed folder
                feature_path = safe_path_join(API_PROCESSED_FOLDER, feature_filename)
                success_feature = cv2.imwrite(feature_path, feature_image)
                
                if success_feature:
                    feature_files[feature_name] = feature_filename
                    feature_images_b64[feature_name] = image_to_base64(feature_image)
                    print(f"Saved API {feature_name} highlighted image with mesh: {feature_path}")
                else:
                    print(f"Failed to save API {feature_name} highlighted image for face {face_id}")
            
            if success_original:
                print(f"Saved API original face {face_id} image: {original_path}")
                saved_faces.append({
                    'face_id': face_id,
                    'original_image': image_to_base64(face_data['original']),
                    'original_filename': original_filename,
                    'feature_images': feature_images_b64,
                    'feature_files': feature_files
                })
            else:
                print(f"Failed to save API original face {face_id} image")
                
    except Exception as e:
        print(f"Error saving API extracted faces: {str(e)}")
    
    return saved_faces

def save_api_rating_images(rating_data, filename_base):
    """Save rating highlighted images for API with mesh highlighting"""
    try:
        ensure_api_directories()
        
        for face in rating_data['individual_ratings']:
            if 'highlighted_image' in face:
                # The highlighted image filename from the rating analysis
                highlighted_filename = face['highlighted_image']
                
                # The image should already be saved in API_PROCESSED_FOLDER
                dest_path = safe_path_join(API_PROCESSED_FOLDER, highlighted_filename)
                
                if os.path.exists(dest_path):
                    print(f"Rating image with mesh already saved: {dest_path}")
                else:
                    print(f"Warning: Rating image not found at {dest_path}")
                    
    except Exception as e:
        print(f"Error saving API rating images: {str(e)}")

@api_app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Main API endpoint for face analysis with mesh highlighting"""
    try:
        # Ensure directories exist
        ensure_api_directories()
        
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify(create_api_response(
                success=False,
                message="No JSON data provided",
                error_code="INVALID_REQUEST"
            )), 400
        
        # Validate required parameters
        image_url = data.get('image_url')
        tool_type = data.get('tool_type', 'face_analysis')
        
        if not image_url:
            return jsonify(create_api_response(
                success=False,
                message="image_url parameter is required",
                error_code="MISSING_IMAGE_URL"
            )), 400
        
        if tool_type not in ['face_analysis', 'face_rating']:
            return jsonify(create_api_response(
                success=False,
                message="tool_type must be 'face_analysis' or 'face_rating'",
                error_code="INVALID_TOOL_TYPE"
            )), 400
        
        print(f"API Request - Image URL: {image_url}, Tool: {tool_type}")
        
        # Download image from URL
        try:
            filepath, filename = download_image_from_url(image_url)
        except ValueError as e:
            return jsonify(create_api_response(
                success=False,
                message=str(e),
                error_code="IMAGE_DOWNLOAD_ERROR"
            )), 400
        except Exception as e:
            return jsonify(create_api_response(
                success=False,
                message=f"Failed to download image: {str(e)}",
                error_code="IMAGE_DOWNLOAD_ERROR"
            )), 500
        
        # Generate filename base for results
        filename_base = filename.rsplit('.', 1)[0]
        
        # Perform analysis based on tool type
        try:
            if tool_type == 'face_analysis':
                # Perform comprehensive face analysis with mesh highlighting
                processed_image, face_count, analysis_results, extracted_faces = comprehensive_face_analysis(filepath)
                
                if processed_image is not None and face_count > 0:
                    # Save extracted faces using API-specific function with mesh
                    saved_faces = save_api_extracted_faces(extracted_faces, filename_base)
                    
                    # Save analysis results to API folder
                    analysis_filename = f"{filename_base}_analysis.json"
                    analysis_path = safe_path_join(API_ANALYSIS_FOLDER, analysis_filename)
                    with open(analysis_path, 'w') as f:
                        json.dump(analysis_results, f, indent=2)
                    
                    # Generate public URLs for images
                    public_urls = generate_public_urls(filename_base, saved_faces, tool_type)
                    
                    # Prepare response data
                    response_data = {
                        'tool_type': tool_type,
                        'face_count': face_count,
                        'analysis_results': analysis_results,
                        'image_urls': public_urls,
                        'filename_base': filename_base,
                        'original_image_url': f"{request.url_root.rstrip('/')}/api/image/{filename}",
                        'mesh_highlighting': True
                    }
                    
                    return jsonify(create_api_response(
                        success=True,
                        message=f"Face analysis with mesh highlighting completed successfully. Found {face_count} face(s).",
                        data=response_data
                    ))
                else:
                    return jsonify(create_api_response(
                        success=False,
                        message="No faces detected in the image",
                        error_code="NO_FACES_DETECTED"
                    )), 400
                    
            elif tool_type == 'face_rating':
                # Perform face rating analysis with mesh highlighting
                rating_result = perform_face_rating_analysis(filepath, filename_base)
                
                if rating_result and rating_result['success']:
                    rating_data = rating_result['data']
                    
                    # Save rating images to API folder (already done in perform_face_rating_analysis)
                    save_api_rating_images(rating_data, filename_base)
                    
                    # Save rating results to API folder
                    rating_filename = f"{filename_base}_rating.json"
                    rating_path = safe_path_join(API_ANALYSIS_FOLDER, rating_filename)
                    with open(rating_path, 'w') as f:
                        json.dump(rating_data, f, indent=2)
                    
                    # Generate public URLs for highlighted images
                    public_urls = {}
                    for face in rating_data['individual_ratings']:
                        face_id = face['face_id']
                        public_urls[f'face_{face_id}'] = {
                            'highlighted': f"{request.url_root.rstrip('/')}/api/image/{face['highlighted_image']}"
                        }
                    
                    # Prepare response data
                    response_data = {
                        'tool_type': tool_type,
                        'face_count': rating_data['face_count'],
                        'average_rating': rating_data['average_rating'],
                        'average_percentage': rating_data['average_percentage'],
                        'highest_rated': rating_data['highest_rated'],
                        'individual_ratings': rating_data['individual_ratings'],
                        'image_urls': public_urls,
                        'filename_base': filename_base,
                        'original_image_url': f"{request.url_root.rstrip('/')}/api/image/{filename}",
                        'mesh_highlighting': True
                    }
                    
                    return jsonify(create_api_response(
                        success=True,
                        message=f"Face rating with mesh highlighting completed successfully. Rated {rating_data['face_count']} face(s).",
                        data=response_data
                    ))
                else:
                    error_msg = rating_result.get('message', 'Face rating analysis failed') if rating_result else 'Face rating analysis failed'
                    return jsonify(create_api_response(
                        success=False,
                        message=error_msg,
                        error_code="RATING_ANALYSIS_FAILED"
                    )), 400
                    
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify(create_api_response(
                success=False,
                message=f"Analysis failed: {str(e)}",
                error_code="ANALYSIS_ERROR"
            )), 500
        
        # Clean up downloaded file
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Warning: Could not clean up file {filepath}: {str(e)}")
            
    except Exception as e:
        print(f"API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify(create_api_response(
            success=False,
            message=f"Internal server error: {str(e)}",
            error_code="INTERNAL_ERROR"
        )), 500

@api_app.route('/api/image/<filename>', methods=['GET'])
def serve_api_image(filename):
    """Serve processed images from API"""
    try:
        # Check in different folders in order of priority
        folders_to_check = [
            API_PROCESSED_FOLDER,      # API processed images
            API_FACES_FOLDER,          # API extracted faces
            API_UPLOAD_FOLDER,         # API uploaded images
        ]
        
        for folder in folders_to_check:
            file_path = safe_path_join(folder, filename)
            if os.path.exists(file_path):
                print(f"Serving image from: {file_path}")
                return send_file(file_path)
        
        # Log all checked paths for debugging
        print(f"Image not found: {filename}")
        print("Checked paths:")
        for folder in folders_to_check:
            file_path = safe_path_join(folder, filename)
            print(f"  - {file_path} (exists: {os.path.exists(file_path)})")
        
        return jsonify(create_api_response(
            success=False,
            message=f"Image not found: {filename}",
            error_code="IMAGE_NOT_FOUND"
        )), 404
        
    except Exception as e:
        print(f"Error serving image {filename}: {str(e)}")
        return jsonify(create_api_response(
            success=False,
            message=f"Error serving image: {str(e)}",
            error_code="IMAGE_SERVE_ERROR"
        )), 500

@api_app.route('/api/download/<filename>', methods=['GET'])
def download_api_file(filename):
    """Download analysis results or images"""
    try:
        # Check in analysis folder first
        file_path = safe_path_join(api_app.config['API_ANALYSIS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        
        # Check in other folders
        folders_to_check = [
            api_app.config['API_FACES_FOLDER'],
            api_app.config['API_PROCESSED_FOLDER'],
            api_app.config['API_UPLOAD_FOLDER']
        ]
        
        for folder in folders_to_check:
            file_path = safe_path_join(folder, filename)
            if os.path.exists(file_path):
                return send_file(file_path, as_attachment=True)
        
        return jsonify(create_api_response(
            success=False,
            message=f"File not found: {filename}",
            error_code="FILE_NOT_FOUND"
        )), 404
        
    except Exception as e:
        print(f"Error downloading file {filename}: {str(e)}")
        return jsonify(create_api_response(
            success=False,
            message=f"Error downloading file: {str(e)}",
            error_code="FILE_DOWNLOAD_ERROR"
        )), 500

@api_app.route('/api/docs', methods=['GET'])
def api_documentation():
    """API documentation endpoint"""
    docs = {
        "title": "Face Analysis API with Mesh Highlighting",
        "version": "1.0.0",
        "description": "API for comprehensive face analysis and rating with detailed facial mesh highlighting",
        "base_url": request.url_root.rstrip('/'),
        "features": [
            "Comprehensive facial feature analysis",
            "Attractiveness rating with detailed breakdown",
            "Full facial mesh highlighting like professional face mapping",
            "White mesh grid overlay with colored feature highlights",
            "Support for multiple faces in single image",
            "RESTful JSON API with CORS support"
        ],
        "endpoints": {
            "health_check": {
                "url": "/api/health",
                "method": "GET",
                "description": "Check API health status"
            },
            "analyze": {
                "url": "/api/analyze",
                "method": "POST",
                "description": "Analyze faces in an image with mesh highlighting",
                "parameters": {
                    "image_url": {
                        "type": "string",
                        "required": True,
                        "description": "URL of the image to analyze"
                    },
                    "tool_type": {
                        "type": "string",
                        "required": False,
                        "default": "face_analysis",
                        "options": ["face_analysis", "face_rating"],
                        "description": "Type of analysis to perform"
                    }
                },
                "example_request": {
                    "image_url": "https://example.com/image.jpg",
                    "tool_type": "face_analysis"
                }
            },
            "get_image": {
                "url": "/api/image/<filename>",
                "method": "GET",
                "description": "Get processed image with mesh highlighting by filename"
            },
            "download_file": {
                "url": "/api/download/<filename>",
                "method": "GET",
                "description": "Download analysis results or images"
            }
        },
        "response_format": {
            "success": "boolean - Whether the request was successful",
            "message": "string - Human readable message",
            "timestamp": "string - ISO timestamp of the response",
            "data": "object - Response data (when successful)",
            "error_code": "string - Error code (when failed)"
        },
        "mesh_highlighting": {
            "description": "All processed images include detailed facial mesh highlighting",
            "features": [
                "White mesh grid mapping facial structure",
                "Green highlights for eyes",
                "Red highlights for mouth/lips",
                "White landmark points at key facial features",
                "Horizontal and vertical structural lines",
                "Diagonal cross-connections for full mesh coverage"
            ]
        },
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size": f"{MAX_FILE_SIZE // (1024 * 1024)}MB",
        "rate_limits": "No rate limits currently implemented",
        "authentication": "No authentication required"
    }
    
    return jsonify(docs)

@api_app.errorhandler(404)
def api_not_found(e):
    """Handle 404 errors"""
    return jsonify(create_api_response(
        success=False,
        message="API endpoint not found",
        error_code="ENDPOINT_NOT_FOUND"
    )), 404

@api_app.errorhandler(405)
def api_method_not_allowed(e):
    """Handle 405 errors"""
    return jsonify(create_api_response(
        success=False,
        message="HTTP method not allowed for this endpoint",
        error_code="METHOD_NOT_ALLOWED"
    )), 405

@api_app.errorhandler(413)
def api_file_too_large(e):
    """Handle file too large errors"""
    return jsonify(create_api_response(
        success=False,
        message=f"File too large. Maximum size allowed: {MAX_FILE_SIZE // (1024 * 1024)}MB",
        error_code="FILE_TOO_LARGE"
    )), 413

@api_app.errorhandler(500)
def api_internal_error(e):
    """Handle internal server errors"""
    return jsonify(create_api_response(
        success=False,
        message="Internal server error occurred",
        error_code="INTERNAL_SERVER_ERROR"
    )), 500

if __name__ == '__main__':
    print("Starting Face Analysis API Server with Mesh Highlighting...")
    print("API Features:")
    print("- Face Analysis: Comprehensive facial feature detection and analysis")
    print("- Face Rating: Attractiveness scoring with detailed breakdown")
    print("- Mesh Highlighting: Full facial mesh overlay like professional face mapping")
    print("- Image URL Support: Download and analyze images from URLs")
    print("- RESTful API: JSON request/response format")
    print("- CORS Enabled: Cross-origin requests supported")
    print(f"- Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"- Max file size: {MAX_FILE_SIZE // (1024 * 1024)}MB")
    print("")
    print("Mesh Highlighting Features:")
    print("- White mesh grid mapping facial structure")
    print("- Green highlights for eyes")
    print("- Red highlights for mouth/lips")
    print("- White landmark points at key facial features")
    print("- Horizontal, vertical, and diagonal structural lines")
    print("- Full mesh coverage for professional face mapping")
    print("")
    print("API Endpoints:")
    print("- GET  /api/health - Health check")
    print("- POST /api/analyze - Main analysis endpoint with mesh highlighting")
    print("- GET  /api/image/<filename> - Serve processed images with mesh")
    print("- GET  /api/download/<filename> - Download files")
    print("- GET  /api/docs - API documentation")
    print("")
    print("Example usage:")
    print("curl -X POST http://localhost:5001/api/analyze \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"image_url\": \"https://example.com/photo.jpg\", \"tool_type\": \"face_analysis\"}'")
    print("")
    
    # Ensure all directories exist at startup
    ensure_api_directories()
    
    api_app.run(debug=True, host='0.0.0.0', port=5002)
