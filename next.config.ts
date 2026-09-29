import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The gallery uses plain <img> tags pointing at object URLs / Drive URLs,
  // so no next/image remote-pattern config is required.
  webpack: (config) => {
    // transformers.js: keep Node-only deps out of the browser bundle.
    config.resolve = config.resolve || {};
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      sharp$: false,
      "onnxruntime-node$": false,
    };
    return config;
  },
};

export default nextConfig;
