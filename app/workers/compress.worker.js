// workers/compress.worker.js
import { compressImageRGB } from '../Compression';

self.onmessage = async (event) => {
  const { R, G, B } = event.data;

  try {
    const result = compressImageRGB(R, G, B);
    self.postMessage({ success: true, data: result }, [result.buffer]);
  } catch (err) {
    self.postMessage({ success: false, error: err.message });
  }
};
