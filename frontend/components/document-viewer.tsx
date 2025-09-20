"use client"

import React from "react"
import { Card } from "./ui/card"
import { FileText, Download } from "lucide-react"
import { Button } from "./ui/button"
import type { Document } from "../app/page"

interface DocumentViewerProps {
  document: Document | null
  highlightPage?: number
}

export function DocumentViewer({ document, highlightPage }: DocumentViewerProps) {
  if (!document) {
    return (
      <Card className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-center text-gray-500">
          <FileText className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <p className="text-lg font-medium">No document selected</p>
          <p className="text-sm">Click on a citation to view a document</p>
        </div>
      </Card>
    )
  }

  const isPDF = document.file_type === 'pdf'
  const isWordDoc = document.file_type === 'docx' || document.file_type === 'doc'

  return (
    <Card className="h-full flex flex-col">
      {/* Document Header */}
      <div className="p-4 border-b bg-white rounded-t-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <FileText className="w-5 h-5 text-blue-600" />
            <div>
              <h3 className="font-semibold text-gray-900 truncate max-w-md">
                {document.document_title}
              </h3>
              <p className="text-sm text-gray-500">
                {document.file_type.toUpperCase()} • {document.chunk_count} chunks
                {highlightPage && ` • Page ${highlightPage}`}
              </p>
            </div>
          </div>
          <Button variant="outline" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        </div>
      </div>

      {/* Document Content */}
      <div className="flex-1 p-4 overflow-auto">
        {isPDF ? (
          <div className="h-full">
            {/* PDF Viewer - For uploaded files, we'll show a placeholder */}
            {document.document_url.startsWith('uploaded://') ? (
              <div className="h-full flex items-center justify-center bg-gray-50 rounded">
                <div className="text-center">
                  <FileText className="w-16 h-16 mx-auto mb-4 text-red-500" />
                  <p className="text-lg font-medium text-gray-700 mb-2">PDF Document</p>
                  <p className="text-sm text-gray-500 mb-4">
                    {document.document_title}
                  </p>
                  {highlightPage && (
                    <p className="text-sm text-blue-600 font-medium mb-4">
                      📍 Referenced Page: {highlightPage}
                    </p>
                  )}
                  <p className="text-xs text-gray-400 mb-4">
                    PDF preview not available for uploaded files. Download to view the full document.
                  </p>
                  <Button variant="outline" size="sm">
                    <Download className="w-4 h-4 mr-2" />
                    Download PDF
                  </Button>
                </div>
              </div>
            ) : (
              <iframe
                src={`${document.document_url}${highlightPage ? `#page=${highlightPage}` : ''}`}
                className="w-full h-full border-0 rounded"
                title={document.document_title}
              />
            )}
          </div>
        ) : isWordDoc ? (
          <div className="h-full flex items-center justify-center bg-gray-50 rounded">
            <div className="text-center">
              <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p className="text-gray-600 mb-2">Word Document Preview</p>
              <p className="text-sm text-gray-500 mb-4">
                {document.document_title}
              </p>
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Download to View
              </Button>
            </div>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center bg-gray-50 rounded">
            <div className="text-center">
              <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p className="text-gray-600 mb-2">Document Preview</p>
              <p className="text-sm text-gray-500 mb-4">
                {document.document_title} ({document.file_type.toUpperCase()})
              </p>
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Download to View
              </Button>
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}