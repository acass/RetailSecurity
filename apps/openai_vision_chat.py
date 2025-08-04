import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
from dotenv import load_dotenv
from openai import OpenAI
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Load environment variables
load_dotenv()

class ImageCaptureProcessor(VideoProcessorBase):
    """Video processor that captures frames for OpenAI analysis."""
    
    def __init__(self):
        self.latest_frame = None
        self.capture_requested = False
        
    def request_capture(self):
        """Request capture of the next frame."""
        self.capture_requested = True
        
    def get_latest_frame(self):
        """Get the latest captured frame."""
        return self.latest_frame
        
    def recv(self, frame):
        """Process video frame and capture if requested."""
        # Convert av.VideoFrame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Capture frame if requested
        if self.capture_requested:
            self.latest_frame = img.copy()
            self.capture_requested = False
        
        # Return the frame unchanged for display
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def encode_image_to_base64(image_array):
    """Convert OpenCV image array to base64 string for OpenAI API."""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return image_base64


def analyze_image_with_openai(image_base64, query, client):
    """Send image and query to OpenAI Vision API."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use GPT-4 with vision capabilities
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


# Global processor instance to avoid session state issues
_processor_instance = None

def get_processor():
    """Get or create the processor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = ImageCaptureProcessor()
    return _processor_instance

def main():
    st.set_page_config(
        page_title="OpenAI Vision Chat",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 OpenAI Vision Chat")
    st.write("Chat with your video stream using OpenAI's vision capabilities!")
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("Please set your OPENAI_API_KEY in the .env file")
        st.stop()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Stream")
        
        # WebRTC streamer
        ctx = webrtc_streamer(
            key="vision-chat",
            video_processor_factory=get_processor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Capture button
        if st.button("📸 Capture Frame", key="capture_btn"):
            processor = get_processor()
            processor.request_capture()
            st.success("Frame capture requested!")
            
        # Show current frame status
        processor = get_processor()
        if processor.get_latest_frame() is not None:
            st.info("✅ Frame captured! You can now ask questions in the chat.")
            # Show a small preview of the captured frame
            frame_rgb = cv2.cvtColor(processor.get_latest_frame(), cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, width=300, caption="Last Captured Frame")
        else:
            st.warning("No frame captured yet. Click 'Capture Frame' first.")
    
    with col2:
        st.subheader("💬 Chat Interface")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if "image" in message:
                        st.image(message["image"], width=200)
        
        # Chat input
        if prompt := st.chat_input("Ask about the video... (e.g., 'What color is the shirt?', 'Is this a boy or girl?')"):
            # Check if we have a captured frame
            processor = get_processor()
            latest_frame = processor.get_latest_frame()
            
            if latest_frame is not None:
                # Get the latest captured frame
                frame = latest_frame
                
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Process the image with OpenAI
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing image..."):
                        try:
                            # Convert frame to base64
                            image_base64 = encode_image_to_base64(frame)
                            
                            # Get response from OpenAI
                            response = analyze_image_with_openai(image_base64, prompt, client)
                            
                            # Display response
                            st.write(response)
                            
                            # Display the analyzed image
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, width=200, caption="Analyzed Image")
                            
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                            response = f"Error occurred: {str(e)}"
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response if 'response' in locals() else "Error occurred",
                    "image": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                })
                
            else:
                with st.chat_message("assistant"):
                    st.warning("Please capture a frame first by clicking the 'Capture Frame' button!")
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 Instructions")
        st.write("""
        1. **Start Video**: Click the video toggle to start your camera
        2. **Capture Frame**: Click 'Capture Frame' to take a snapshot
        3. **Ask Questions**: Type your question in the chat input
        
        **Example Questions:**
        - "What color is the shirt in this image?"
        - "Is this a boy or girl?"
        - "How many people are in the image?"
        - "What objects can you see?"
        - "Describe what's happening in this scene"
        - "What is the person wearing?"
        """)
        
        st.header("⚙️ Settings")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.header("ℹ️ About")
        st.write("""
        This app uses OpenAI's GPT-4 with vision capabilities to analyze 
        images captured from your video stream. Ask natural language 
        questions about what you see!
        """)


if __name__ == "__main__":
    main()