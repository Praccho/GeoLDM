const puppeteer = require('puppeteer');
const fs = require('fs');
const https = require('https');

async function captureMapImage(lat, lng) {
    // Generate the image URL
    const imageUrl = `https://maps.googleapis.com/maps/api/staticmap?center=${lat},${lng}&zoom=19&size=600x400&maptype=satellite&key=AIzaSyBYjvDRRKVHdWJM-0gkaRilP8CaNG6DG9M`;

    // Define where you want to save the image, ensuring you use the correct path and filename
    const savePath = `C:\\Users\\seoli\\testing images\\map_${lat}_${lng}.png`;

    // Download and save the image
    await downloadImage(imageUrl, savePath)
        .then(() => console.log(`Image saved to ${savePath}`))
        .catch(err => console.error(`Error saving image: ${err.message}`));
}

function downloadImage(url, savePath) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(savePath);
        https.get(url, (response) => {
            response.pipe(file);
            file.on('finish', () => {
                file.close();
                resolve();
            });
        }).on('error', (err) => {
            fs.unlink(savePath); // Delete the file async on error
            reject(err);
        });
    });
}

// Example coordinates
const coords = [
    { lat: 28.45765, lng: -16.363564 }, // First set of coordinates
    { lat: 50.23435, lng: -70.123940 }  // Second set of coordinates
];

coords.forEach(coord => {
    captureMapImage(coord.lat, coord.lng).then(() => console.log('Done'));
});
