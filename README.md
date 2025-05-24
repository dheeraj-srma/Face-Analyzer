# 🧠 Real-Time Age, Gender & Emotion Detection using OpenCV and DeepFace

This project uses OpenCV and DeepFace to detect human faces in real-time from a webcam or image and analyze their **age**, **gender**, and **dominant emotion**. It's fast, lightweight, and works on CPU without the need for GPU acceleration.

---

## 🚀 Features

* 🎯 **Real-Time Face Detection** using OpenCV's DNN face detector.
* 🧬 **Age & Gender Prediction** using DeepFace.
* 😊 **Emotion Recognition** with high accuracy.
* 📷 **Webcam Support** and optional image input via command line.
* ✨ Clean, readable code with suppressed model logs for a smooth experience.

---

## 📃 Requirements

* Python 3.7+
* OpenCV (`opencv-python`)
* DeepFace
* NumPy

```bash
pip install opencv-python deepface numpy
```

---

## 📂 How to Use

1. **Download the model files** (see instructions below).
2. **Run the script** with optional image path:

```bash
python detect.py --image your_image.jpg
```

Or just launch the webcam:

```bash
python detect.py
```

---

## 📁 Model Files

Place these in the root folder:

* `opencv_face_detector.pbtxt`
* `opencv_face_detector_uint8.pb`

📅 Download from OpenCV:

* [pbtxt file](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/opencv_face_detector.pbtxt)
* [pb file](https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel)

Rename the `.caffemodel` to:
`opencv_face_detector_uint8.pb`

---

## 📸 Output Example

Live feed window with bounding boxes and labels like:

```
[Man, 27, happy]
```

---

## 📌 Todo & Ideas

* [ ] Add face recognition for attendance
* [ ] Create a GUI for toggling input/video
* [ ] Export analysis data to CSV or dashboard
* [ ] Add multi-face support & tracking

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT License](LICENSE)
