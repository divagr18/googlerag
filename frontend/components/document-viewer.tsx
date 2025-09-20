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
                                {highlightPage && (
                                    <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs font-medium">
                                        📍 Page {highlightPage}
                                    </span>
                                )}
                            </p>
                        </div>
                    </div>
                    <Button variant="outline" size="sm" onClick={() => {
                        const downloadUrl = document.document_url.startsWith('uploaded://')
                            ? `http://localhost:8000/api/v1/hackrx/file/${document.document_id}`
                            : document.document_url;
                        window.open(downloadUrl, '_blank');
                    }}>
                        <Download className="w-4 h-4 mr-2" />
                        Download
                    </Button>
                </div>
            </div>

            {/* Document Content */}
            <div className="flex-1 p-4 overflow-auto">
                {isPDF ? (
                    <div className="h-full">
                        {/* PDF Viewer - For uploaded files, serve them through the API */}
                        {document.document_url.startsWith('uploaded://') ? (
                            <iframe
                                key={`${document.document_id}-${highlightPage || 'default'}`}
                                src={`http://localhost:8000/api/v1/hackrx/file/${document.document_id}${highlightPage ? `#page=${highlightPage}` : ''}`}
                                className="w-full h-full border-0 rounded"
                                title={document.document_title}
                            />
                        ) : (
                            <iframe
                                key={`${document.document_id}-${highlightPage || 'default'}`}
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