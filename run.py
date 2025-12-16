import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import time
import datetime
import os
import threading
import numpy as np

# =============================================================================
# CLASS 1: IMAGE PROCESSOR
# This is where all the computer vision magic lives
# =============================================================================
class ImageProcessor:
    def __init__(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
        except:
            print("Heads up: could not load cascade files.")

        # Simple background subtractor
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=40
        )

    def detect_faces(self, frame):
        # Face detector works better on grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw a box + label around every face we find
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                frame, "Face", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
            )
        return frame, len(faces)

    def detect_eyes(self, frame):
        # Same idea: detect eyes on a gray copy of the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 3)

        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

    def detect_motion(self, frame):
        # Treat the first frames as "background", then highlight whatever moves
        mask = self.background_subtractor.apply(frame)

        # Clean up the mask so tiny noise doesn’t count as motion
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False
        for contour in contours:
            # Ignore super small blobs
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                motion_detected = True

        if motion_detected:
            cv2.putText(
                frame, "MOTION DETECTED", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
            )
        return frame

    def apply_canny(self, frame):
        # Classic edge detector – shows outlines of stuff in the scene
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def track_blue_color(self, frame):
        # Switch to HSV – way nicer for color‑based stuff
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rough range for blue-ish pixels
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Build a mask and keep only the blue parts
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        return res

    def apply_sepia(self, frame):
        # Quick sepia filter using a 3x3 color transform matrix
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        return cv2.transform(frame, kernel)


# =============================================================================
# CLASS 2: MAIN APPLICATION
# Handles the GUI, camera loop, buttons
# =============================================================================
class AdvancedCVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Computer Vision System - Team 8")
        self.root.geometry("1200x800")

        # Shared state for the whole app
        self.processor = ImageProcessor()
        self.cap = None
        self.is_running = False
        self.current_filter = "Normal"
        self.is_recording = False
        self.video_writer = None
        self.current_frame = None

        # Build all screens/widgets
        self.setup_ui()

    def setup_ui(self):
        # Notebook = tabbed interface (Camera tab + About tab)
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True)

        # Main camera tab
        self.tab_camera = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_camera, text='  Live Vision Studio  ')
        self.setup_camera_tab()

        # Credits / team info tab
        self.tab_about = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_about, text='  About Team  ')
        self.setup_about_tab()

        # Simple status bar at the bottom of the window
        self.status_var = tk.StringVar()
        self.status_var.set("System Ready")
        self.status_bar = tk.Label(
            self.root, textvariable=self.status_var,
            bd=1, relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_camera_tab(self):
        # Big container: video on the left, controls on the right
        main_layout = tk.Frame(self.tab_camera, bg="#2c3e50")
        main_layout.pack(fill='both', expand=True)

        # ==========================
        # LEFT SIDE: VIDEO FEED
        # ==========================
        # Create a frame specifically for the video to force it to expand
        video_container = tk.Frame(main_layout, bg="black")
        video_container.pack(side=tk.LEFT, fill='both', expand=True, padx=10, pady=10)

        # The label inside needs to expand to fill the container
        self.video_frame = tk.Label(video_container, bg="black")
        self.video_frame.pack(fill='both', expand=True)

        # ==========================
        # RIGHT SIDE: CONTROL PANEL
        # ==========================
        # Fixed width for the sidebar so it doesn't jump around
        control_panel = tk.Frame(main_layout, bg="#ecf0f1", width=250)
        control_panel.pack(side=tk.RIGHT, fill='y', padx=10, pady=10)
        control_panel.pack_propagate(False) # Force the frame to stay at width=250

        # Panel title
        tk.Label(
            control_panel, text="Control Panel",
            font=("Helvetica", 16, "bold"),
            bg="#ecf0f1", fg="#2c3e50"
        ).pack(pady=20)

        # Start / Stop camera buttons
        btn_frame = tk.Frame(control_panel, bg="#ecf0f1")
        btn_frame.pack(fill='x', pady=5)

        tk.Button(
            btn_frame, text="▶ START", command=self.start_camera,
            bg="#27ae60", fg="white", font=("Arial", 10, "bold"), width=10
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame, text="⏹ STOP", command=self.stop_camera,
            bg="#c0392b", fg="white", font=("Arial", 10, "bold"), width=10
        ).pack(side=tk.RIGHT, padx=5)

        # Filter options (radio buttons)
        tk.Label(
            control_panel, text="Algorithms & Filters",
            font=("Helvetica", 12, "bold"), bg="#ecf0f1"
        ).pack(pady=(20, 5))

        self.filter_var = tk.StringVar(value="Normal")
        filters = [
            ("Normal Feed", "Normal"),
            ("Face Detection", "Face"),
            ("Eye Detection", "Eyes"),
            ("Motion Detection", "Motion"),
            ("Edge Detection (Canny)", "Canny"),
            ("Blue Color Tracking", "Color"),
            ("Sepia Filter", "Sepia"),
            ("Gaussian Blur", "Blur"),
            ("Invert Mode", "Invert")
        ]

        # One radio button per filter mode
        for text, val in filters:
            tk.Radiobutton(
                control_panel, text=text,
                variable=self.filter_var, value=val,
                command=self.change_filter,
                bg="#ecf0f1", font=("Arial", 11), anchor='w'
            ).pack(fill='x', padx=20, pady=2)

        # Screenshot / Recording buttons
        tk.Label(
            control_panel, text="Media Tools",
            font=("Helvetica", 12, "bold"), bg="#ecf0f1"
        ).pack(pady=(30, 5))

        tk.Button(
            control_panel, text="📷 Take Screenshot",
            command=self.take_screenshot, bg="#2980b9", fg="white"
        ).pack(fill='x', padx=20, pady=5)

        self.btn_record = tk.Button(
            control_panel, text="⏺ Start Recording",
            command=self.toggle_recording, bg="#e67e22", fg="white"
        )
        self.btn_record.pack(fill='x', padx=20, pady=5)

    def setup_about_tab(self):
        # Simple static page to show team + supervisor
        frame = tk.Frame(self.tab_about, bg="white")
        frame.pack(fill='both', expand=True)

        tk.Label(
            frame, text="Computer Vision Project",
            font=("Helvetica", 24, "bold"), bg="white"
        ).pack(pady=40)

        tk.Label(
            frame, text="Judged by: Dr. Fatma Elsayed",
            font=("Helvetica", 16), bg="white"
        ).pack(pady=10)

        tk.Label(
            frame, text="Team Members:",
            font=("Helvetica", 14, "bold"), bg="white"
        ).pack(pady=20)

        members = [
            "1. David Ashraf - Motion Analysis & Background Subtraction",
            "2. Hamza Hasanain - Face Detection (Haar Cascades)",
            "3. John Shawky - Eye Feature Extraction",
            "4. Mohammed Naser - Color Space Segmentation (HSV)",
            "5. Youssef Ragab - Structural Analysis (Edge Detection)",
            "6. Sugood Salama - Image Transformation (Kernels)",
            "7. Amin Mohammed - Pre-processing & Noise Reduction",
            "8. Eslam Mohammed - Video Stream Management"
        ]

        for m in members:
            tk.Label(frame, text=m, font=("Arial", 12), bg="white").pack()

    # ========================
    # App logic / event handlers
    # ========================
    def change_filter(self):
        # Just update the current filter mode when user clicks a radio button
        self.current_filter = self.filter_var.get()
        self.status_var.set(f"Filter changed to: {self.current_filter}")

    def start_camera(self):
        if not self.is_running:
            # 0 = default webcam
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            # Run the camera loop on a background thread so the GUI doesn’t freeze
            threading.Thread(
                target=self.process_video_stream, daemon=True
            ).start()

    def stop_camera(self):
        # Stop the loop and clean up stuff
        self.is_running = False
        if self.is_recording:
            self.toggle_recording()
        if self.cap:
            self.cap.release()
        self.video_frame.config(image="")

    def toggle_recording(self):
        # Start/stop recording the processed video to a file
        if not self.is_running:
            messagebox.showwarning("Warning", "Start the camera first!")
            return

        if not self.is_recording:
            # Start recording
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            filename = f'video_{datetime.datetime.now().strftime("%H%M%S")}.avi'
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, 20.0, (640, 480)
            )
            self.is_recording = True
            self.btn_record.config(text="⏹ Stop Recording", bg="red")
            self.status_var.set(f"Recording started: {filename}")
        else:
            # Stop recording
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
            self.btn_record.config(text="⏺ Start Recording", bg="#e67e22")
            self.status_var.set("Recording saved.")

    def take_screenshot(self):
        # Save the last displayed frame as a PNG
        if self.current_frame is not None:
            filename = f'snap_{datetime.datetime.now().strftime("%H%M%S")}.png'
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Saved", f"Screenshot saved as {filename}")

    def process_video_stream(self):
        # Main camera loop that keeps grabbing frames until we stop
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            final_frame = frame.copy()
            face_count = 0

            # Route through the chosen CV algorithm
            if self.current_filter == "Face":
                final_frame, face_count = self.processor.detect_faces(final_frame)
                if face_count > 0:
                    self.status_var.set(f"Faces detected: {face_count}")

            elif self.current_filter == "Eyes":
                final_frame = self.processor.detect_eyes(final_frame)
            elif self.current_filter == "Motion":
                final_frame = self.processor.detect_motion(final_frame)
            elif self.current_filter == "Canny":
                final_frame = self.processor.apply_canny(final_frame)
            elif self.current_filter == "Color":
                final_frame = self.processor.track_blue_color(final_frame)
            elif self.current_filter == "Sepia":
                final_frame = self.processor.apply_sepia(final_frame)
            elif self.current_filter == "Blur":
                final_frame = cv2.GaussianBlur(final_frame, (15, 15), 0)
            elif self.current_filter == "Invert":
                final_frame = cv2.bitwise_not(final_frame)

            # If recording is on, also push frames into the video writer
            if self.is_recording and self.video_writer:
                rec_frame = cv2.resize(final_frame, (640, 480))
                self.video_writer.write(rec_frame)

            # Keep a copy around for screenshots
            self.current_frame = final_frame

            # Draw it on the GUI
            self.display_frame(final_frame)

            # Tiny sleep so CPU doesn’t go crazy
            time.sleep(0.01)

    def display_frame(self, frame):
        # Resize the frame to fit inside the label while keeping aspect ratio
        display_width = self.video_frame.winfo_width()
        display_height = self.video_frame.winfo_height()

        if display_width > 10 and display_height > 10:
            h, w, _ = frame.shape
            ratio = min(display_width / w, display_height / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            frame = cv2.resize(frame, (new_w, new_h))

        # Convert from BGR (OpenCV) to RGB (for Tkinter)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))

        # Must keep a reference or Python garbage collects the image
        self.video_frame.imgtk = img_tk
        self.video_frame.configure(image=img_tk)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    # Make an output folder to drop videos/images there later
    if not os.path.exists("media_output"):
        os.makedirs("media_output")

    root = tk.Tk()
    app = AdvancedCVApp(root)
    root.mainloop()
