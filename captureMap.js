const puppeteer = require('puppeteer');
const fs = require('fs');
const https = require('https');

async function captureMapImage(lat, lng) {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();

    // Modify the URL to point to the location of your satellite.html file
    await page.goto('../satellite.html');

    // Optionally, simulate interactions with the page if needed
    // For example, to click a button to start the drawing process
    // await page.click('#yourButtonId');

    // Wait for the map to load and the image URL to be generated
    // This might involve waiting for a specific element to be visible
    // await page.waitForSelector('#staticLink', {visible: true});

    // Assuming the image URL is now available in an element's attribute
    // const imageUrl = await page.$eval('#staticLink', el => el.href);

    // Alternatively, if you can generate the image URL directly

    // Download and save the image using the image URL
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
    // Generate the image URL
    const imageUrl = `https://maps.googleapis.com/maps/api/staticmap?center=${lat},${lng}&zoom=19&size=600x400&maptype=satellite&key=AIzaSyA48hnkfq4JTGeu4ZxaDOUBRc4lhw19bB4`;

    // Define where you want to save the image
    //const savePath = `./map_images/map_${lat}_${lng}.png`;
    const savePath = "C:\Users\seoli\testing images";
    // Download and save the image
    await downloadImage(imageUrl, savePath)
        .then(() => console.log(`Image saved to ${savePath}`))
        .catch(err => console.error(`Error saving image: ${err.message}`));

    // Close the browser
    await browser.close();
    // Example: Output the generated URL (you would replace this with download logic)
}

// Example coordinates
const lat = 28.45765;
const lng = -16.363564;

captureMapImage(lat, lng).then(() => console.log('Done'));
