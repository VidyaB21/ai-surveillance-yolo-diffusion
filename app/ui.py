import streamlit as st
import os
import subprocess

st.title("🔍 AI Surveillance System")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)

    video_path = os.path.join("data", uploaded_file.name)

    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("Video uploaded!")
    st.video(video_path)

    if st.button("Run AI System"):
        st.write("Processing... please wait ⏳")

        subprocess.run(["python", "app/main_pipeline.py", video_path])

        st.success("Done!")

        st.subheader("Generated Images")

        for file in sorted(os.listdir("outputs")):
            if file.endswith(".png"):
                st.image(f"outputs/{file}")