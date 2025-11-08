import os
import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from detection_service import DetectionService, Detection


class QueryProcessor:
    """Processes natural language queries about object detection results."""
    
    def __init__(self, detection_service: DetectionService):
        """Initialize query processor.
        
        Args:
            detection_service: DetectionService instance for getting current detections
        """
        self.detection_service = detection_service
        self.client = self._initialize_openai_client()
        
    def _initialize_openai_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI client with API key from environment."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
            return None
            
        try:
            return OpenAI(api_key=api_key)
        except Exception as e:
            print(f"Warning: Failed to initialize OpenAI client: {e}")
            print("Natural language queries will not be available")
            return None
        
def process_query(self, query: str, camera_name: str) -> Dict[str, Any]:
        """Process a natural language query about detections from a specific camera.
        
        Args:
            query: Natural language query (e.g., "Are there any dogs?")
            camera_name: The name of the camera source
            
        Returns:
            Dict with query results and metadata
        """
        if not self.client:
            return {
                "success": False,
                "error": "OpenAI client not initialized. Check OPENAI_API_KEY.",
                "answer": "AI service unavailable"
            }
            
        current_detections = self.detection_service.get_current_detections(camera_name)
        detection_summary = self.detection_service.get_detection_summary(camera_name)
        
        available_classes = self.detection_service.get_available_classes()
        
        if not current_detections:
            context = f"On camera '{camera_name}', no objects are currently detected."
        else:
            class_counts = detection_summary.get("class_counts", {})
            context_parts = [f"{count} {name}{'s' if count > 1 else ''}" for name, count in class_counts.items()]
            context = f"On camera '{camera_name}', detected objects are: {', '.join(context_parts)}."
            
        # Create AI prompt
        system_prompt = f"""You are an AI assistant analyzing a real-time video surveillance feed.

Available object types that can be detected: {', '.join(available_classes)}

Current detection context: {context}

Your job is to answer the user's question about what's currently visible in the video stream.
Be specific, accurate, and conversational. If asked about objects not currently detected, clearly state they are not visible.
If asked about object counts, provide exact numbers.
If the user asks about synonyms (e.g., "people" for "person", "cars" for "car"), understand the intent.

Answer format should be natural and conversational, as if you're watching the video alongside the user."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "detection_context": {
                    "total_detections": detection_summary["total_detections"],
                    "unique_classes": detection_summary["unique_classes"],
                    "class_counts": detection_summary["class_counts"],
                    "last_updated": detection_summary["last_updated"]
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "answer": f"I encountered an error processing your query: {str(e)}"
            }
            
    def extract_object_classes(self, query: str) -> List[str]:
        """Extract object class names from a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            List of relevant object class names mentioned in the query
        """
        if not self.client:
            return []
            
        available_classes = self.detection_service.get_available_classes()
        
        system_prompt = f"""Extract object class names from the user's query.
Available classes: {', '.join(available_classes)}

Return ONLY a JSON list of exact class names from the available list that are mentioned or implied in the query.
Handle synonyms appropriately (e.g., "people" -> "person", "cars" -> "car", "bikes" -> "bicycle").

Examples:
- "Are there any dogs?" -> ["dog"]
- "Show me people and cars" -> ["person", "car"]  
- "Any bottles or cups?" -> ["bottle", "cup"]
- "Is there a TV?" -> ["tv"]

Return empty list [] if no relevant classes are found."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                class_list = json.loads(result)
                if isinstance(class_list, list):
                    return [cls for cls in class_list if cls in available_classes]
            except json.JSONDecodeError:
                # Fallback: try to extract list from text
                match = re.search(r'\[(.*?)\]', result)
                if match:
                    items = match.group(1).split(',')
                    class_list = [
                        item.strip().strip('"').strip("'") 
                        for item in items if item.strip()
                    ]
                    return [cls for cls in class_list if cls in available_classes]
            
            return []
            
        except Exception as e:
            print(f"Error extracting object classes: {e}")
            return []
            
    def get_query_suggestions(self) -> List[str]:
        """Get example queries based on current detections.
        
        Returns:
            List of suggested natural language queries
        """
        current_detections = self.detection_service.get_current_detections()
        
        base_suggestions = [
            "What do you see in the video?",
            "Are there any people?",
            "How many objects are detected?",
            "What animals are visible?",
            "Are there any vehicles?",
            "Is anyone carrying a bag?",
            "Are there any electronic devices?",
        ]
        
        if current_detections:
            detected_classes = list(set(d.class_name for d in current_detections))
            specific_suggestions = [
                f"How many {cls}s do you see?" for cls in detected_classes[:3]
            ]
            specific_suggestions.extend([
                f"Is there a {cls} in the video?" for cls in detected_classes[:2]
            ])
            return specific_suggestions + base_suggestions[:4]
        
        return base_suggestions
        
    def analyze_detection_trends(self, time_window_seconds: int = 60) -> Dict[str, Any]:
        """Analyze detection patterns over time (placeholder for future implementation).
        
        Args:
            time_window_seconds: Time window for analysis
            
        Returns:
            Dict with trend analysis results
        """
        # This would require storing detection history over time
        # For now, return current state
        return {
            "message": "Trend analysis not yet implemented",
            "current_summary": self.detection_service.get_detection_summary(),
            "time_window": time_window_seconds
        }