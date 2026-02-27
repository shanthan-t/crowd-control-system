const puppeteer = require('puppeteer');
(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  page.on('requestfailed', request => {
    if (request.failure()) console.log('PAGE REQUEST FAILED:', request.url(), request.failure().errorText);
  });
  await page.goto('http://localhost:5173/');
  await new Promise(r => setTimeout(r, 2000));
  await page.evaluate(() => {
    const el = Array.from(document.querySelectorAll('*')).find(e => e.textContent === 'Spatial Intelligence');
    if (el) el.click();
  });
  await new Promise(r => setTimeout(r, 3000));
  await browser.close();
})();
