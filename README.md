# Smart-Visual-System-for-Visually-Impaired-People-Raspberry-Pi-4-B-

Clone using the web URL
https://github.com/dadkarim/Smart-Visual-System-for-Visually-Impaired-People-Raspberry-Pi-4-B-.git

Project Introduction:
This project develops three auditory interaction methods to help visually impaired people perceive their surroundings. Using a VR headset with a camera and ultrasonic sensors, it captures environmental data and delivers it via obstacle detection, image-to-speech, and path sonification as speech and sound cues.

Background:
Built on real research — grounded in academic findings, not just guesswork.
The white cane reimagined — traditionally a tool for balance and navigation, it's the oldest companion for the blind and partially sighted.
Early electronic aids were basic — first-gen gadgets used ultrasonic sensors, warning users with simple buzzes and beeps.
Old solutions wore blinders too — each prior technology solved just one narrow problem, leaving the bigger picture incomplete.
Books required the internet — earlier reading aids depended on mobile apps that demanded a constant online connection.
One creative workaround — researchers strapped three acoustic sensors to a belt and paired it with a vibrating bracelet to silently alert users of nearby obstacles.

Objectives:
Object Detection & Recognition — Identify surrounding objects in real time and communicate them to visually impaired users through clear, natural speech output.
Assistive Book Reading — Enable visually impaired individuals to independently read books by converting printed text into spoken audio.
Collision Prevention — Measure distances to nearby obstacles accurately, helping users navigate safely and avoid potential collisions.

Hardware Setup:
A VR box integrates a camera, Raspberry Pi, headphones, and four ultrasonic sensors. The camera covers a 60° horizontal and 40° vertical field of view — sufficient for most real-world navigation scenarios.

Software Stack:
The system runs on Raspbian OS, programmed via Thonny Python and accessed remotely through VNC. Key libraries include OpenCV, Tesseract OCR, pyttsx3, NumPy, and TensorFlow, with SSDLite + MobileNetV2 powering object detection.

Three Functional Modes:
Each mode is activated by a dedicated button on the device:
Button 1 — Detection & Recognition: Captures images via PiCam, runs SSDLite with MobileNetV2 and OpenCV to identify objects, and announces them through speech.
Button 2 — Reading Mode: Uses PiCam to capture text-containing images, then processes them with Tesseract OCR, NumPy, and pyttsx3 to read the content aloud.
Button 3 — Navigation Mode: Four ultrasonic sensors cover left, right, and forward directions, while a fourth is angled at 45° to detect ditches and low-lying obstacles, guiding safe movement.

Open raspberry-pi-setup-guide for complete installation.
[raspberry-pi-setup-guide.md](https://github.com/user-attachments/files/28562387/raspberry-pi-setup-guide.md)
