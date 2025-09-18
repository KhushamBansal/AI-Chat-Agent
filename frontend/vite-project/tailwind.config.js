/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'chat-user': '#3b82f6',
        'chat-bot': '#6366f1',
        'chat-system': '#f59e0b',
      },
    },
  },
  plugins: [],
}