#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SceneValidator: A tool to validate scene composition against style guidelines
Version: 0.1
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser("~/.scene_validator/logs/scene_validator.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("scene_validator")

# Import optional dependencies
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning("Gemini API not available. Some features will be limited.")
    GEMINI_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    logger.warning("OpenCV not available. Frame extraction will be limited.")
    CV2_AVAILABLE = False


class SceneValidator:
    """Main class for validating scenes against style guidelines."""
    
    def __init__(self, guidelines_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the SceneValidator.
        
        Args:
            guidelines_path: Path to the style guidelines JSON file
            config_path: Path to the configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.guidelines = self._load_guidelines(guidelines_path)
        self._setup_gemini_api()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        default_config = {
            "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
            "default_guidelines": os.path.expanduser("~/.scene_validator/guidelines/default.json"),
            "output_directory": os.path.expanduser("~/.scene_validator/output"),
            "log_level": "INFO"
        }
        
        if not config_path:
            config_path = os.path.expanduser("~/.scene_validator/config.json")
            
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info(f"Configuration loaded from {config_path}")
            else:
                logger.warning(f"Configuration file not found at {config_path}, using defaults")
                
            # Create output directory if it doesn't exist
            os.makedirs(default_config["output_directory"], exist_ok=True)
            
            # Set logging level
            logger.setLevel(getattr(logging, default_config["log_level"]))
            
            return default_config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return default_config
    
    def _load_guidelines(self, guidelines_path: Optional[str]) -> Dict[str, Any]:
        """Load style guidelines from JSON file."""
        if not guidelines_path:
            guidelines_path = self.config["default_guidelines"]
            
        try:
            if os.path.exists(guidelines_path):
                with open(guidelines_path, 'r') as f:
                    guidelines = json.load(f)
                    logger.info(f"Guidelines loaded from {guidelines_path}")
                return guidelines
            else:
                logger.error(f"Guidelines file not found at {guidelines_path}")
                raise FileNotFoundError(f"Guidelines file not found: {guidelines_path}")
        except Exception as e:
            logger.error(f"Error loading guidelines: {str(e)}")
            raise
    
    def _setup_gemini_api(self) -> None:
        """Configure Gemini API with credentials."""
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini API module not available")
            return
            
        try:
            api_key = self.config.get("gemini_api_key", "")
            if not api_key:
                logger.warning("No Gemini API key provided in configuration")
                return
                
            genai.configure(api_key=api_key)
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {str(e)}")
    
    def extract_frames(self, scene_path: str, interval: float = 1.0) -> List[Dict[str, Any]]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            scene_path: Path to the video file
            interval: Interval in seconds between extracted frames
            
        Returns:
            List of dictionaries containing frame data and metadata
        """
        if not CV2_AVAILABLE:
            logger.error("OpenCV is required for frame extraction")
            return []
            
        frames = []
        try:
            # Open the video file
            video = cv2.VideoCapture(scene_path)
            if not video.isOpened():
                logger.error(f"Could not open video file: {scene_path}")
                return []
                
            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            logger.info(f"Video properties: {fps} FPS, {frame_count} frames, {duration:.2f} seconds")
            
            # Calculate frame interval
            frame_interval = int(fps * interval)
            if frame_interval < 1:
                frame_interval = 1
                
            # Extract frames
            frame_index = 0
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                    
                if frame_index % frame_interval == 0:
                    # Convert frame to RGB (from BGR)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Calculate timestamp
                    timestamp = frame_index / fps
                    
                    # Store frame data
                    frames.append({
                        "index": frame_index,
                        "timestamp": timestamp,
                        "frame": rgb_frame,
                        "height": rgb_frame.shape[0],
                        "width": rgb_frame.shape[1]
                    })
                    
                    logger.debug(f"Extracted frame at {timestamp:.2f}s")
                
                frame_index += 1
                
            video.release()
            logger.info(f"Extracted {len(frames)} frames from {scene_path}")
            
            return frames
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            return []
    
    def analyze_scene(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze scene frames using Gemini API.
        
        Args:
            frames: List of frame dictionaries from extract_frames()
            
        Returns:
            List of issues detected in the frames
        """
        if not GEMINI_AVAILABLE or not frames:
            return []
            
        issues = []
        
        try:
            # Configure Gemini model
            model = genai.GenerativeModel('gemini-pro-vision')
            
            # Process frames in batches (to avoid API limits)
            batch_size = min(5, len(frames))
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i+batch_size]
                
                # Prepare batch for analysis
                for frame_data in batch_frames:
                    # Convert NumPy array to PIL Image for Gemini API
                    from PIL import Image
                    frame = frame_data["frame"]
                    pil_image = Image.fromarray(frame)
                    
                    # Create prompt for Gemini
                    prompt = f"""
                    Analyze this frame from a video scene and identify any issues with:
                    1. Composition (rule of thirds, framing, balance)
                    2. Color grading (consistency, saturation, contrast)
                    3. Lighting (exposure, key light, fill light, shadows)
                    
                    Style guidelines to consider:
                    {json.dumps(self.guidelines, indent=2)}
                    
                    Provide specific issues in JSON format with the following structure:
                    {{
                      "issues": [
                        {{
                          "type": "composition|color|lighting",
                          "severity": "high|medium|low",
                          "description": "detailed description of the issue",
                          "recommendation": "specific recommendation to fix"
                        }}
                      ]
                    }}
                    """
                    
                    # Call Gemini API
                    response = model.generate_content([prompt, pil_image])
                    
                    # Extract JSON from response
                    try:
                        result_text = response.text
                        # Find JSON block in response
                        json_start = result_text.find('{')
                        json_end = result_text.rfind('}') + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = result_text[json_start:json_end]
                            result = json.loads(json_str)
                            
                            # Add frame reference to each issue
                            frame_reference = f"frame_{frame_data['index']:05d}_{frame_data['timestamp']:.2f}s"
                            for issue in result.get("issues", []):
                                issue["frame_reference"] = frame_reference
                                issues.append(issue)
                    except Exception as e:
                        logger.error(f"Error parsing Gemini response: {str(e)}")
                        continue
            
            logger.info(f"Analysis complete: {len(issues)} issues found")
            return issues
        except Exception as e:
            logger.error(f"Error analyzing scene: {str(e)}")
            return []
    
    def calculate_compliance(self, issues: List[Dict[str, Any]]) -> float:
        """
        Calculate overall compliance score based on issues.
        
        Args:
            issues: List of issues from analyze_scene()
            
        Returns:
            Compliance score (0-100)
        """
        if not issues:
            return 100.0
            
        # Weight issues by severity
        severity_weights = {
            "high": 10.0,
            "medium": 5.0,
            "low": 2.0
        }
        
        # Calculate penalty points
        total_penalty = sum(severity_weights.get(issue.get("severity", "low"), 2.0) for issue in issues)
        
        # Cap penalty at 100 points
        total_penalty = min(total_penalty, 100)
        
        # Calculate compliance score
        compliance_score = 100.0 - total_penalty
        
        return max(0.0, compliance_score)
    
    def generate_report(self, validation_results: Dict[str, Any], format: str = "json") -> str:
        """
        Generate formatted report from validation results.
        
        Args:
            validation_results: Validation results dictionary
            format: Output format (json, txt, html)
            
        Returns:
            Formatted report string
        """
        if format.lower() == "json":
            return json.dumps(validation_results, indent=2)
        elif format.lower() == "txt":
            # Generate plain text report
            report = []
            report.append(f"Scene Validation Report")
            report.append(f"=====================")
            report.append(f"Scene ID: {validation_results.get('scene_id', 'Unknown')}")
            report.append(f"Timestamp: {validation_results.get('validation_timestamp', 'Unknown')}")
            report.append(f"Overall Compliance: {validation_results.get('overall_compliance', 0):.1f}%")
            report.append(f"")
            report.append(f"Issues:")
            
            for i, issue in enumerate(validation_results.get("issues", []), 1):
                report.append(f"")
                report.append(f"Issue #{i} - {issue.get('type', 'Unknown')} - Severity: {issue.get('severity', 'Unknown')}")
                report.append(f"Description: {issue.get('description', 'No description')}")
                report.append(f"Frame: {issue.get('frame_reference', 'Unknown')}")
                if "recommendation" in issue:
                    report.append(f"Recommendation: {issue.get('recommendation', '')}")
            
            report.append(f"")
            report.append(f"Summary: {validation_results.get('summary', '')}")
            
            return "\n".join(report)
        elif format.lower() == "html":
            # Generate HTML report
            html = []
            html.append("<!DOCTYPE html>")
            html.append("<html>")
            html.append("<head>")
            html.append("  <title>Scene Validation Report</title>")
            html.append("  <style>")
            html.append("    body { font-family: Arial, sans-serif; margin: 20px; }")
            html.append("    h1 { color: #333; }")
            html.append("    .compliance { font-size: 24px; font-weight: bold; }")
            html.append("    .high { color: #d9534f; }")
            html.append("    .medium { color: #f0ad4e; }")
            html.append("    .low { color: #5bc0de; }")
            html.append("    .issue { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }")
            html.append("  </style>")
            html.append("</head>")
            html.append("<body>")
            html.append(f"  <h1>Scene Validation Report</h1>")
            html.append(f"  <p>Scene ID: {validation_results.get('scene_id', 'Unknown')}</p>")
            html.append(f"  <p>Timestamp: {validation_results.get('validation_timestamp', 'Unknown')}</p>")
            html.append(f"  <p class='compliance'>Overall Compliance: {validation_results.get('overall_compliance', 0):.1f}%</p>")
            html.append(f"  <h2>Issues</h2>")
            
            for issue in validation_results.get("issues", []):
                severity = issue.get('severity', 'low')
                html.append(f"  <div class='issue'>")
                html.append(f"    <h3 class='{severity}'>{issue.get('type', 'Unknown')} Issue - Severity: {severity}</h3>")
                html.append(f"    <p><strong>Description:</strong> {issue.get('description', 'No description')}</p>")
                html.append(f"    <p><strong>Frame:</strong> {issue.get('frame_reference', 'Unknown')}</p>")
                if "recommendation" in issue:
                    html.append(f"    <p><strong>Recommendation:</strong> {issue.get('recommendation', '')}</p>")
                html.append(f"  </div>")
            
            html.append(f"  <h2>Summary</h2>")
            html.append(f"  <p>{validation_results.get('summary', '')}</p>")
            html.append("</body>")
            html.append("</html>")
            
            return "\n".join(html)
        else:
            logger.error(f"Unsupported format: {format}")
            return json.dumps(validation_results, indent=2)
    
    def validate(self, scene_path: str, guidelines_path: Optional[str] = None, 
                output_format: str = "json", tolerance_level: float = 80.0,
                include_recommendations: bool = True) -> Dict[str, Any]:
        """
        Main validation function to process a scene against guidelines.
        
        Args:
            scene_path: Path to the scene file or directory
            guidelines_path: Path to guidelines JSON (overrides instance guidelines)
            output_format: Format for output (json, txt, html)
            tolerance_level: Threshold for flagging issues (0-100)
            include_recommendations: Whether to include AI recommendations
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Starting validation of scene: {scene_path}")
        
        # Load guidelines if specified
        if guidelines_path:
            self.guidelines = self._load_guidelines(guidelines_path)
        
        try:
            # Extract scene ID from path
            scene_id = os.path.splitext(os.path.basename(scene_path))[0]
            
            # Extract frames from scene
            frames = self.extract_frames(scene_path)
            if not frames:
                raise ValueError(f"No frames could be extracted from scene: {scene_path}")
            
            # Analyze scene
            issues = self.analyze_scene(frames)
            
            # Calculate compliance score
            compliance_score = self.calculate_compliance(issues)
            
            # Filter issues by severity if tolerance level is set
            if tolerance_level < 100:
                severity_threshold = "medium" if tolerance_level >= 70 else "low"
                filtered_issues = [issue for issue in issues if 
                                  self._severity_to_value(issue.get("severity", "low")) >= 
                                  self._severity_to_value(severity_threshold)]
                issues = filtered_issues
            
            # Remove recommendations if not requested
            if not include_recommendations:
                for issue in issues:
                    if "recommendation" in issue:
                        del issue["recommendation"]
            
            # Generate summary
            summary = self._generate_summary(compliance_score, issues)
            
            # Prepare validation results
            validation_results = {
                "scene_id": scene_id,
                "validation_timestamp": datetime.now().isoformat(),
                "overall_compliance": compliance_score,
                "issues": issues,
                "summary": summary
            }
            
            # Generate output report
            report = self.generate_report(validation_results, output_format)
            
            # Save report to output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{scene_id}_validation_{timestamp}.{output_format}"
            output_path = os.path.join(self.config["output_directory"], output_filename)
            
            with open(output_path, 'w') as f:
                f.write(report)
                
            logger.info(f"Validation report saved to: {output_path}")
            
            return validation_results
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise
    
    def _severity_to_value(self, severity: str) -> int:
        """Convert severity string to numeric value for comparison."""
        severity_values = {
            "high": 3,
            "medium": 2,
            "low": 1
        }
        return severity_values.get(severity.lower(), 0)
    
    def _generate_summary(self, compliance_score: float, issues: List[Dict[str, Any]]) -> str:
        """Generate a human-readable summary of validation results."""
        if compliance_score >= 90:
            quality = "excellent"
        elif compliance_score >= 75:
            quality = "good"
        elif compliance_score >= 50:
            quality = "fair"
        else:
            quality = "poor"
            
        # Count issues by type
        issue_counts = {}
        for issue in issues:
            issue_type = issue.get("type", "unknown")
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
        # Generate summary text
        if not issues:
            return f"The scene passed validation with {quality} compliance ({compliance_score:.1f}%). No issues were detected."
        else:
            issue_text = ", ".join(f"{count} {issue_type}" for issue_type, count in issue_counts.items())
            return f"The scene achieved {quality} compliance ({compliance_score:.1f}%). Found {len(issues)} issues: {issue_text}."


def main():
    """Command-line interface for SceneValidator."""
    parser = argparse.ArgumentParser(description="Validate scene composition against style guidelines")
    parser.add_argument("--scene", required=True, help="Path to the scene file")
    parser.add_argument("--guidelines", help="Path to style guidelines JSON")
    parser.add_argument("--config", help="Path to configuration JSON")
    parser.add_argument("--output-format", choices=["json", "txt", "html"], default="txt", 
                        help="Output format (default: txt)")
    parser.add_argument("--tolerance", type=float, default=80.0, 
                        help="Tolerance level for issues (0-100, default: 80)")
    parser.add_argument("--include-recommendations", action="store_true", 
                        help="Include AI recommendations in output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a scene")
    validate_parser.add_argument("--scene", required=True, help="Path to the scene file")
    validate_parser.add_argument("--guidelines", help="Path to style guidelines JSON")
    
    # Serve command (for web interface)
    serve_parser = subparsers.add_parser("serve", help="Start web interface")
    serve_parser.add_argument("--port", type=int, default=5000, help="Port for web server")
    serve_parser.add_argument("--host", default="localhost", help="Host for web server")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Initialize validator
        validator = SceneValidator(
            guidelines_path=args.guidelines, 
            config_path=args.config
        )
        
        if args.command == "serve":
            try:
                from flask import Flask, request, jsonify, render_template
                app = Flask(__name__)
                
                @app.route('/')
                def index():
                    return "SceneValidator Web Interface"
                
                @app.route('/validate', methods=['POST'])
                def validate_endpoint():
                    if 'file' not in request.files:
                        return jsonify({"error": "No file provided"}), 400
                    
                    file = request.files['file']
                    if file.filename == '':
                        return jsonify({"error": "No file selected"}), 400
                    
                    # Save uploaded file temporarily
                    temp_path = os.path.join("/tmp", file.filename)
                    file.save(temp_path)
                    
                    # Process validation
                    results = validator.validate(
                        scene_path=temp_path,
                        output_format="json",
                        tolerance_level=float(request.form.get('tolerance', 80.0)),
                        include_recommendations=request.form.get('include_recommendations') == 'true'
                    )
                    
                    # Clean up
                    os.remove(temp_path)
                    
                    return jsonify(results)
                
                # Start web server
                print(f"Starting web interface at http://{args.host}:{args.port}")
                app.run(host=args.host, port=args.port)
            except ImportError:
                logger.error("Flask is required for web interface. Install with 'pip install flask'")
                sys.exit(1)
        else:
            # Default to validate command
            if not args.scene:
                logger.error("Scene path is required")
                parser.print_help()
                sys.exit(1)
                
            results = validator.validate(
                scene_path=args.scene,
                guidelines_path=args.guidelines,
                output_format=args.output_format,
                tolerance_level=args.tolerance,
                include_recommendations=args.include_recommendations
            )
            
            # Print results to console
            if args.output_format == "json":
                print(json.dumps(results, indent=2))
            else:
                print(f"Validation complete: {results['overall_compliance']:.1f}% compliance")
                print(f"Found {len(results['issues'])} issues")
                print(f"Report saved to: {validator.config['output_directory']}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()