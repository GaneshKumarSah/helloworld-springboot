const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const faceapi = require('face-api.js');
const canvas = require('canvas');

const app = express();
const port = 3000;
app.use(express.json());

const cors = require('cors');
app.use(cors());

const MODEL_PATH = path.join(__dirname, 'models');
const UPLOADS_DIR = path.join(__dirname, 'uploads');
const TEMP_DIR = path.join(UPLOADS_DIR, 'temp');

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// In-memory database
const usersDB = [];

// Ensure necessary directories exist
if (!fs.existsSync(UPLOADS_DIR)) fs.mkdirSync(UPLOADS_DIR);
if (!fs.existsSync(TEMP_DIR)) fs.mkdirSync(TEMP_DIR);

// Load Face-API models
async function loadModels() {
    await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_PATH);
    await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_PATH);
    await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH);
}
loadModels();

// Multer storage setup for /upload
const uploadStorage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOADS_DIR),
    filename: (req, file, cb) => cb(null, file.originalname)
});
const upload = multer({ storage: uploadStorage });

// Multer memory storage for /compare
const tempUpload = multer({ storage: multer.memoryStorage() });

// Upload API
app.post('/upload', upload.single('image'), (req, res) => {
    if (!req.file || !req.body.username) return res.status(400).send('Missing file or username.');
    
    usersDB.push({
        username: req.body.username,
        filename: req.file.originalname,
        status_active: true
    });
    res.json({ message: 'File uploaded successfully!' });
});

// Activate or Deactivate user API
app.post('/update-status', (req, res) => {
    const { username, status_active } = req.body;
    const user = usersDB.find(u => u.username === username);
    if (!user) return res.status(404).send('User not found.');
    user.status_active = status_active;
    res.json({ message: `User ${username} status updated.` });
});

// Fetch stored users API
app.get('/users', (req, res) => {
    res.json(usersDB);
});

// Compare Faces Function
async function compareFaces(imagePath1, imagePath2) {
    const img1 = await canvas.loadImage(imagePath1);
    const img2 = await canvas.loadImage(imagePath2);

    const face1 = await faceapi.detectSingleFace(img1).withFaceLandmarks().withFaceDescriptor();
    const face2 = await faceapi.detectSingleFace(img2).withFaceLandmarks().withFaceDescriptor();

    if (!face1 || !face2) return null;
    const distance = faceapi.euclideanDistance(face1.descriptor, face2.descriptor);
    return (1 - distance) * 100;
}

// Compare API
app.post('/compare', tempUpload.single('image'), async (req, res) => {
    if (!req.file) return res.status(400).send('No image uploaded.');

    // Save file temporarily in /uploads/temp
    const tempFilename = `${uuidv4()}-${req.file.originalname}`;
    const tempFilePath = path.join(TEMP_DIR, tempFilename);
    fs.writeFileSync(tempFilePath, req.file.buffer);

    // Fetch only active users
    const activeUsers = usersDB.filter(user => user.status_active);

    let comparisons = [];
    for (const user of activeUsers) {
        const imageToComparePath = path.join(UPLOADS_DIR, user.filename);
        if (!fs.existsSync(imageToComparePath)) continue;
        try {
            const similarity = await compareFaces(tempFilePath, imageToComparePath);
            if (similarity !== null) {
                comparisons.push({ filename: user.filename, username: user.username, similarity });
            }
        } catch (error) {
            console.error(`Error comparing ${tempFilename} with ${user.filename}:`, error);
        }
    }

    // Delete temp file after comparison
    if (fs.existsSync(tempFilePath)) fs.unlinkSync(tempFilePath);

    if (comparisons.length === 0) {
        return res.json({ loginAllowed: false, username: null });
    }

    comparisons.sort((a, b) => b.similarity - a.similarity);
    const bestMatch = comparisons[0];

    if (bestMatch.similarity > 70) {
        return res.json({ loginAllowed: true, username: bestMatch.username });
    } else if (bestMatch.similarity >= 50) {
        return res.json({ loginAllowed: false, username: bestMatch.username });
    }

    return res.json({ loginAllowed: false, username: null });
});

// Start server
app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});

