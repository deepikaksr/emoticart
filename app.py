import streamlit as st
import cv2
import numpy as np
import pandas as pd
from detect_utils import detect_objects_and_emotions

# Session State Initialization
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "prev_emotion" not in st.session_state:
    st.session_state.prev_emotion = None
if "prev_product" not in st.session_state:
    st.session_state.prev_product = None
if "saved_positive_product" not in st.session_state:
    st.session_state.saved_positive_product = None
if "exploration_handled" not in st.session_state:
    st.session_state.exploration_handled = False
if "explore_ui_shown" not in st.session_state:
    st.session_state.explore_ui_shown = False
if "show_product_details" not in st.session_state:
    st.session_state.show_product_details = False
if "no_response" not in st.session_state:
    st.session_state.no_response = False

# Page Configuration and Styling
st.set_page_config(layout="wide", page_title="EmotiCart", page_icon="🛒")
st.markdown("""
    <style>
    /* Main Styling */
    .main { background-color: #f8f9fa; padding: 1.5rem; }
    
    /* Header Styling */
    .header {
        font-size: 2.8em;
        font-weight: 700;
        text-align: center;
        color: #2e7d32;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        padding: 10px;
        background: linear-gradient(to right, #e8f5e9, #c8e6c9);
        border-radius: 12px;
    }
    
    /* Camera Feed Styling */
    .camera-feed { border-radius: 12px; background-color: transparent; }
    .stImage img { width: 100%; border-radius: 12px; }
    
    /* Suggestion Box Styling */
    .suggestion-box {
        background: linear-gradient(to bottom right, #e3f2fd, #bbdefb);
        border-left: 5px solid #1e88e5;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .suggestion-title { color: #1565c0; font-size: 1.5em; margin-bottom: 10px; font-weight: 600; }
    .suggestion-content { color: #0d47a1; font-size: 1.1em; line-height: 1.5; }
    .emotion-highlight { background-color: #c8e6c9; color: #2e7d32; padding: 2px 8px; border-radius: 8px; font-weight: 600; }
    .product-highlight { background-color: #bbdefb; color: #1565c0; padding: 2px 8px; border-radius: 8px; font-weight: 600; }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(to bottom right, #43a047, #2e7d32);
        color: white;
        padding: 12px 20px;
        font-size: 1.1em;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        background: linear-gradient(to bottom right, #2e7d32, #1b5e20);
    }
    
    /* Status Container */
    .status-container {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        text-align: center;
        border-left: 5px solid #43a047;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Instructions Panel */
    .instructions {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        font-size: 0.95em;
        color: #e65100;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 30px;
        padding-top: 10px;
        border-top: 1px solid #e0e0e0;
        color: #757575;
        font-size: 0.9em;
    }
    
    /* Product Card */
    .product-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    }
    
    /* Exploration Question Section */
    .explore-question {
        background: linear-gradient(to right, #e8f5e9, #c8e6c9);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        margin: 20px auto;
        max-width: 80%;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .explore-question h3 {
        color: #2e7d32;
        margin-bottom: 15px;
        font-weight: 600;
    }
    .explore-question p {
        color: #1b5e20;
        font-size: 1.1em;
        margin-bottom: 25px;
    }
    
    /* Center aligned buttons container */
    .center-buttons {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 0 auto;
        max-width: 60%;
    }
    
    /* Yes button style */
    .yes-btn {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* No button style */
    .no-btn {
        background-color: #f5f5f5 !important;
        color: #424242 !important;
        border: 1px solid #bdbdbd !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'>🛒 EmotiCart: Smart Shopping Experience 🔍</div>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; margin-bottom: 25px; color: #555;">
        Discover products that match your mood with our emotion-sensing technology!
    </div>
""", unsafe_allow_html=True)

# Layout Setup
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h3 style='text-align: center; color: #2e7d32;'>📹 Live Camera Feed</h3>", unsafe_allow_html=True)
    # Camera control button placement
    cam_col1, cam_col2, cam_col3 = st.columns([1, 2, 1])
    with cam_col2:
        camera_btn_text = "▶️ Start Camera" if not st.session_state.camera_on else "⏹️ Stop Camera"
        if st.button(camera_btn_text, key="toggle_cam", help="Toggle Camera"):
            st.session_state.camera_on = not st.session_state.camera_on
            st.session_state.show_product_details = False  # Reset product display when toggling

    FRAME_WINDOW = st.empty()  # For live camera feed
    
    # Status indicator when the camera is off
    if not st.session_state.camera_on:
        st.markdown("""
            <div class="status-container">
                <p>📸 Camera is currently off. Press the Start Camera button to begin!</p>
            </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='text-align: center; color: #1565c0;'>🧠 Smart Suggestions</h3>", unsafe_allow_html=True)
    suggestion_placeholder = st.empty()
    if not st.session_state.camera_on:
        suggestion_placeholder.markdown("""
            <div class="suggestion-box">
                <h4 class="suggestion-title">🔮 Ready to Help!</h4>
                <p class="suggestion-content">We'll show personalized product suggestions based on your emotions when the camera is turned on.</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div class="instructions">
            <h4>📝 How It Works:</h4>
            <ol>
                <li>Start the camera.</li>
                <li>We detect your emotions and the product you are viewing.</li>
                <li>If you're in a positive mood (happy, neutral, or curious), we save that product.</li>
                <li>Then you can choose to explore more about it.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# Create a center container for exploration UI and product details
center_container = st.container()
explore_ui = center_container.empty()      
product_details = center_container.empty()   

# Product Display Function
def display_product_details(product_category):
    explore_ui.empty() 
    df = pd.read_csv('product_data.csv')
    filtered_products = df[df['Category'].str.lower() == product_category.lower()]
    
    if not filtered_products.empty:
        st.subheader(f"🏷️ Products in Category: {product_category.title()}")
        for index, row in filtered_products.iterrows():
            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    category_lower = product_category.lower()
                    if "book" in category_lower:
                        emoji = "📚"
                    elif "food" in category_lower or "fruit" in category_lower:
                        emoji = "🍎"
                    elif "drink" in category_lower or "bottle" in category_lower:
                        emoji = "🥤"
                    elif "clothing" in category_lower or "shirt" in category_lower:
                        emoji = "👕"
                    elif "electronic" in category_lower or "phone" in category_lower:
                        emoji = "📱"
                    elif "chair" in category_lower or "furniture" in category_lower:
                        emoji = "🪑"
                    else:
                        emoji = "🛍️"
                    st.markdown(f"""
                        <div style="background-color: #f5f5f5; 
                                    border-radius: 10px; 
                                    height: 150px; 
                                    display: flex; 
                                    align-items: center; 
                                    justify-content: center; 
                                    font-size: 50px;">
                            {emoji}
                        </div>
                    """, unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"<h3 style='color: #1565c0; margin-bottom: 5px;'>{row['Title']}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color: #2e7d32; margin-top: 0; margin-bottom: 10px;'>₹{row['Price (INR)']}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: #555;'>{row['Description']}</p>", unsafe_allow_html=True)
                st.markdown("<hr style='margin: 20px 0; opacity: 0.3;'>", unsafe_allow_html=True)
    else:
        st.info(f"No products found in the category: {product_category}")

# Main Camera Loop and Processing
if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Camera could not be opened. Please check permissions and try again.")
    else:
        try:
            while st.session_state.camera_on and not st.session_state.show_product_details:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to capture frame. Please check your camera connection.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_frame = frame_rgb.copy()

                # Detect objects and emotions from the frame
                objects, emotion = detect_objects_and_emotions(frame_rgb)
                filtered_objects = [obj for obj in objects if obj.lower() != "person"]

                info_bar_height = 40
                info_bar = np.ones((info_bar_height, display_frame.shape[1], 3), dtype=np.uint8) * 240
                emotion_text = emotion if emotion else "Scanning..."
                objects_text = ", ".join(filtered_objects[:3]) if filtered_objects else "No products"
                combined_frame = np.vstack((display_frame, info_bar))
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(combined_frame, f"😊 Emotion: {emotion_text}", (10, display_frame.shape[0] + 25), font, 0.7, (33,33,33), 2)
                cv2.putText(combined_frame, f"🛍️ Products: {objects_text}", (combined_frame.shape[1]-300, display_frame.shape[0]+25), font, 0.6, (33,33,33), 2)
                FRAME_WINDOW.image(combined_frame, channels="RGB")

                # Update smart suggestions
                if emotion and filtered_objects:
                    product = filtered_objects[0]
                    if (emotion != st.session_state.prev_emotion) or (product != st.session_state.prev_product):
                        st.session_state.prev_emotion = emotion
                        st.session_state.prev_product = product

                        # Determine an emoji for the detected emotion
                        emotion_lower = emotion.lower()
                        emotion_emoji = "😊"
                        if emotion_lower == "happy":
                            emotion_emoji = "😄"
                        elif emotion_lower == "sad":
                            emotion_emoji = "😢"
                        elif emotion_lower == "angry":
                            emotion_emoji = "😠"
                        elif emotion_lower == "surprised":
                            emotion_emoji = "😲"
                        elif emotion_lower == "neutral":
                            emotion_emoji = "😐"
                        elif emotion_lower == "disgust":
                            emotion_emoji = "🤢"
                        elif emotion_lower == "fear":
                            emotion_emoji = "😨"
                        
                        # Determine product emoji based on product type
                        product_emoji = "🛍️"
                        if "book" in product.lower():
                            product_emoji = "📚"
                        elif "food" in product.lower() or "fruit" in product.lower():
                            product_emoji = "🍎"
                        elif "drink" in product.lower() or "bottle" in product.lower():
                            product_emoji = "🥤"
                        elif "clothing" in product.lower() or "shirt" in product.lower():
                            product_emoji = "👕"
                        elif "electronic" in product.lower() or "phone" in product.lower():
                            product_emoji = "📱"
                        elif "chair" in product.lower() or "furniture" in product.lower():
                            product_emoji = "🪑"
                        
                        suggestion_placeholder.markdown(
                            f"""
                            <div class='suggestion-box'>
                                <h4 class='suggestion-title'>{emotion_emoji} Smart Suggestion {product_emoji}</h4>
                                <p class='suggestion-content'>
                                    You seem <span class='emotion-highlight'>{emotion}</span>!<br><br>
                                    We recommend exploring more <span class='product-highlight'>{product}</span> options!<br><br>
                                    <strong>✨ Perfect match for your current mood! ✨</strong>
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Save the product if the emotion is positive
                        if emotion_lower in ["happy", "neutral", "curious"]:
                            st.session_state.saved_positive_product = product
                            st.session_state.exploration_handled = False
                            st.session_state.explore_ui_shown = False
                else:
                    suggestion_placeholder.markdown("""
                        <div class="suggestion-box">
                            <h4 class="suggestion-title">🔍 Analyzing...</h4>
                            <p class="suggestion-content">Looking for products and analyzing your emotions.</p>
                        </div>
                    """, unsafe_allow_html=True)

                if st.session_state.saved_positive_product and not st.session_state.exploration_handled and not st.session_state.explore_ui_shown:
                    st.session_state.explore_ui_shown = True
                    with explore_ui.container():
                        st.markdown(
                            f"""
                            <div class="explore-question">
                                <h3>🌟 Product Recommendation</h3>
                                <p>You seemed to be in a positive mood regarding <strong>{st.session_state.saved_positive_product}</strong>.<br>
                                Would you like to explore more about the latest {st.session_state.saved_positive_product}?</p>
                            </div>
                            """, unsafe_allow_html=True
                        )
                        st.markdown('<div class="center-buttons">', unsafe_allow_html=True)
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("✅ Yes, Show Me!", key=f"yes_btn_{st.session_state.saved_positive_product}", help="Show product details"):
                                st.session_state.show_product_details = True
                                st.session_state.exploration_handled = True
                                st.session_state.camera_on = False   # Stop the video immediately
                        with col_no:
                            if st.button("❌ No, Thanks", key=f"no_btn_{st.session_state.saved_positive_product}", help="Continue shopping"):
                                st.session_state.no_response = True
                                st.session_state.exploration_handled = True
                                st.session_state.camera_on = False   # Stop the video immediately
                        st.markdown('</div>', unsafe_allow_html=True)
                    # Give a very short delay to let the button click register
                    cv2.waitKey(1)
        finally:
            cap.release()

# Display Outcome After Video Stops
if not st.session_state.camera_on:
    if st.session_state.show_product_details:
        display_product_details(st.session_state.saved_positive_product)
    elif st.session_state.no_response:
        st.markdown("""
            <div class="status-container">
                <p>If you need help or when your mood becomes better, feel free to reach out!</p>
            </div>
        """, unsafe_allow_html=True)
    st.session_state.no_response = False

# Footer
st.markdown("""
    <div class="footer">
        <p>EmotiCart © 2025 | Making your shopping experience more personalized 🛒</p>
    </div>
""", unsafe_allow_html=True)
