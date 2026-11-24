# Hand Gesture Volume Control 🎛️🖐️

Change your computer's volume by moving your hand in front of the webcam. The project tracks your hand in real time and sends volume up and volume down key presses based on which way your hand moves.

## How It Works

1. OpenCV captures frames from your webcam.
2. Each frame is converted to RGB and passed to MediaPipe Hands, which detects up to two hands and returns 21 landmarks per hand.
3. The program takes the tracked landmark's position and converts it to a relative x and y value (0 to 1) so it doesn't depend on camera resolution.
4. It compares the current horizontal position with the previous frame. If the hand moved left in the frame, it presses `volumedown`. Otherwise it presses `volumeup`.
5. PyAutoGUI sends the key press to the operating system, so it works with whatever audio device you're using.

Press `q` in the preview window to quit.

## Tech Stack

Python, OpenCV, MediaPipe, PyAutoGUI

## Getting Started

**Clone the repo**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

**Install dependencies**

```bash
pip install opencv-python mediapipe pyautogui
```

**Run it**

```bash
python volume.py
```

Make sure your webcam is connected and not being used by another app.

## Usage

Hold your hand in view of the camera and move it sideways. Moving left lowers the volume, moving right raises it. The camera preview isn't mirrored, so left and right follow the video frame, not your own left and right.

## Platform Support

Built with PyAutoGUI's media keys, so it's aimed at Windows. Linux may work depending on your desktop environment, and macOS support for volume keys is limited.

## Known Limitations

This is a simple prototype. It reacts to movement direction only, not hand position or a specific gesture, and a key press is sent on every frame, so the volume can change quickly.

## Future Improvements

Pinch-distance control (thumb to index finger) for precise volume levels, a gesture to mute and unmute, on-screen volume display, landmark drawing on the preview, and smoothing or a dead zone to stop jitter.

## License

MIT License

---

Built by [Vaibhav Gupta](https://github.com/Vaibhav0407)
