# Converting YOLO

## Some Commands

To start a video stream reader on the host machine:

```bash
gst-launch-1.0 udpsrc multicast-group=10.42.0.1 port=5000 ! application/x-rtp, payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink
```

Setup the streamer on device:

```Python

class Streamer(object):
    def __init__(self, host='10.42.0.1', w=256, h=256, fps=30, port=5000):
        gst_out = (
            f'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! '
            f'rtph264pay config-interval=1 pt=96 ! udpsink host={host} port={port} auto-multicast=true'
        )
        self.w, self.h = w, h
        self.fps = fps
        self.out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, self.fps, (self.w, self.h), True)

    def __del__(self):
        self.out.release()

    def send(self, frame):
        self.out.write(frame)
```

Read camera input using python:
```Python

class CameraReader(object):
    def __init__(self, device="video2", w=256, h=256, fps=30):
        self.device_path = os.path.join("/dev", device)
        self.cap = cv2.VideoCapture(self.device_path)

        self.h, self.w = h, w
        if not self.cap.isOpened():
            raise IOError(f"Can't open: {self.device_path}")
    
    def __del__(self):
        self.cap.release()
        
    def get_frame(self):
        ret, image = self.cap.read()

        if not ret:
            print("Error: Could not read frame.")

        image = cv2.resize(image, (self.w, self.h), interpolation = cv2.INTER_LINEAR)
        return image
```