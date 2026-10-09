const CACHE_NAME = 'botlens-cache-v5';
const ASSETS = [
  '/static/icon.png',
  '/static/icon-192.png',
  '/static/icon-512.png'
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  // Navigations (index.html) go network-first so app updates land immediately;
  // only the offline fallback uses the cache. Static assets stay cache-first.
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(async () => (await caches.match(event.request)) ||
        new Response('<h2 style="font-family:sans-serif;padding:24px">BOTLens: no connection. It will reload when the internet is back.</h2><script>setTimeout(()=>location.reload(),10000)</script>',
          { status: 503, headers: { 'Content-Type': 'text/html' } }))
    );
    return;
  }
  event.respondWith(
    caches.match(event.request).then((response) => response || fetch(event.request))
  );
});
