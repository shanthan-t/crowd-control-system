import puppeteer from 'puppeteer';

(async () => {
    const browser = await puppeteer.launch({ headless: 'new' });
    const page = await browser.newPage();
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', error => console.log('\n\n=== PAGE ERROR ===\n', error.message, '\n==================\n'));
    page.on('requestfailed', request => {
        if (request.failure()) console.log('PAGE REQUEST FAILED:', request.url(), request.failure().errorText);
    });
    console.log("Navigating...");
    await page.goto('http://localhost:5173/');

    console.log("Setting localStorage injection for Admin login bypass...");
    await page.evaluate(() => {
        localStorage.setItem('sentinel_user', 'admin');
        localStorage.setItem('sentinel_role', 'admin');
        localStorage.setItem('sentinel_token', 'fake-token');
    });

    await page.reload();
    await new Promise(r => setTimeout(r, 2000));

    console.log("Clicking Spatial Intelligence...");
    await page.evaluate(() => {
        const el = Array.from(document.querySelectorAll('*')).find(e => e.textContent === 'Spatial Intelligence');
        if (el) el.click();
        else console.log('Spatial Intelligence button not found');
    });
    await new Promise(r => setTimeout(r, 2000));
    console.log("Closing...");
    await browser.close();
})();
