/** @type {import('next').NextConfig} */
const nextConfig = {
    experimental: {
        esmExternals: false,
    },
    // Enable SWC minification for better performance
    swcMinify: true,
    // Configure API URL for production
    env: {
        NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    },
}

module.exports = nextConfig