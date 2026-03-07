import type { Metadata, Viewport } from 'next'
import { Inter, Space_Mono } from 'next/font/google'

import './globals.css'

const _inter = Inter({ subsets: ['latin'], variable: '--font-inter' })
const _spaceMono = Space_Mono({ weight: ['400', '700'], subsets: ['latin'], variable: '--font-space-mono' })

export const metadata: Metadata = {
  title: "Lady Nisy's CV Regenerator - AI-Powered Resume Builder",
  description: 'Upload your old CV and a job offer to generate a perfectly tailored resume with AI.',
}

export const viewport: Viewport = {
  themeColor: '#2B2D54',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`${_inter.variable} ${_spaceMono.variable} font-sans antialiased`}>
        {children}
      </body>
    </html>
  )
}
