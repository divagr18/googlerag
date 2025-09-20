"use client"

import type React from "react"

import { Button } from "./ui/button"
import { Card } from "./ui/card"
import { Avatar, AvatarFallback } from "./ui/avatar"
import { Upload, FileText, Loader2 } from "lucide-react"
import type { Document } from "@/app/page"
import { useRef, useState, type DragEvent } from "react"

interface DocumentLibraryProps {
  documents: Document[]
  onSelectDocument: (document: Document) => void
  onFileUpload: (files: FileList) => void
}

export function DocumentLibrary({ documents, onSelectDocument, onFileUpload }: DocumentLibraryProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const getScoreColor = (score: number) => {
    if (score >= 80) return "bg-green-500"
    if (score >= 60) return "bg-yellow-500"
    return "bg-red-500"
  }

  const getScoreText = (score: number) => {
    if (score >= 80) return "Low Risk"
    if (score >= 60) return "Medium Risk"
    return "High Risk"
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      onFileUpload(files)
    }
    // Reset input value to allow uploading the same file again
    e.target.value = ""
  }

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      onFileUpload(files)
    }
  }

  return (
    <div
      className={`min-h-screen bg-black text-white transition-colors ${isDragOver ? "bg-zinc-900" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,application/pdf"
        multiple
        onChange={handleFileChange}
        className="hidden"
      />

      {isDragOver && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-zinc-900 border-2 border-dashed border-white/30 rounded-lg p-8 text-center">
            <Upload className="w-12 h-12 text-white mx-auto mb-4" />
            <p className="text-white text-lg font-medium">Drop PDF files here</p>
            <p className="text-zinc-400 text-sm mt-2">Only PDF files are supported</p>
          </div>
        </div>
      )}

      <header className="border-b border-zinc-800 px-4 py-3">
        <div className="flex items-center justify-between max-w-6xl mx-auto">
          <h1 className="text-sm font-medium text-white">Contract Guardian</h1>
          <Avatar className="w-6 h-6">
            <AvatarFallback className="bg-zinc-800 text-white text-xs">CG</AvatarFallback>
          </Avatar>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6">
        <div className="mb-6">
          <Button
            onClick={handleUploadClick}
            className="bg-white hover:bg-zinc-200 text-black text-xs px-3 py-1.5 h-auto"
          >
            <Upload className="w-3 h-3 mr-1.5" />
            Upload
          </Button>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
          {documents.map((document) => (
            <Card
              key={document.document_id}
              className={`relative bg-zinc-900 border-zinc-800 hover:border-white/20 transition-colors cursor-pointer ${
                document.isProcessing ? "opacity-60" : ""
              }`}
              onClick={() => !document.isProcessing && onSelectDocument(document)}
            >
              <div className="p-3">
                {!document.isProcessing && document.guardianScore && (
                  <div
                    className={`absolute -top-1 -right-1 w-5 h-5 rounded-full ${getScoreColor(document.guardianScore)} flex items-center justify-center text-white text-xs font-medium`}
                  >
                    {document.guardianScore}
                  </div>
                )}

                <div className="flex justify-center mb-2">
                  {document.isProcessing ? (
                    <Loader2 className="w-6 h-6 text-zinc-500 animate-spin" />
                  ) : (
                    <FileText className="w-6 h-6 text-zinc-400" />
                  )}
                </div>

                <div className="text-center">
                  <h3 className="font-medium text-white mb-1 text-xs leading-tight line-clamp-2">
                    {document.document_title}
                  </h3>
                  <p className="text-xs text-zinc-500 mb-1">
                    {document.isProcessing ? "Analyzing..." : new Date(document.processed_timestamp).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric", 
                      year: "numeric",
                    })}
                  </p>
                  {!document.isProcessing && document.guardianScore && (
                    <p className="text-xs text-zinc-400">{getScoreText(document.guardianScore)}</p>
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>
      </main>
    </div>
  )
}
