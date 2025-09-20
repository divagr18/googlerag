"use client"

import React, { useState } from "react"
import { Button } from "./ui/button"
import { ArrowLeft } from "lucide-react"
import { ChatInterface } from "./chat-interface"
import { DocumentViewer } from "./document-viewer"
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels"
import type { Document } from "../app/page"

interface TwoColumnChatProps {
    initialDocument: Document
    documents: Document[]
    onBack: () => void
    onFileUpload: (files: FileList) => void
}

export function TwoColumnChat({ initialDocument, documents, onBack, onFileUpload }: TwoColumnChatProps) {
    const [selectedDocument, setSelectedDocument] = useState<Document>(initialDocument)
    const [highlightPage, setHighlightPage] = useState<number | undefined>(undefined)

    const handleSelectDocument = (document: Document, page?: number) => {
        console.log('TwoColumnChat: Selecting document:', document, 'page:', page)
        setSelectedDocument(document)
        setHighlightPage(page)
    }

    const handleCitationClick = (document: Document, page?: number) => {
        console.log('TwoColumnChat: Citation clicked for document:', document, 'page:', page)
        handleSelectDocument(document, page)
    }

    return (
        <div className="h-screen flex flex-col bg-gray-50">
            {/* Header */}
            <div className="flex items-center justify-between p-4 bg-white border-b">
                <Button variant="ghost" onClick={onBack} className="flex items-center">
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Back to Library
                </Button>
                <h1 className="text-lg font-semibold text-gray-900">
                    Document Analysis
                </h1>
                <div></div> {/* Spacer for centering */}
            </div>

            {/* Two Column Layout with Resizable Panels */}
            <div className="flex-1 overflow-hidden">
                <PanelGroup direction="horizontal">
                    {/* Left Panel - Chat Interface */}
                    <Panel defaultSize={50} minSize={30}>
                        <div className="h-full">
                            <ChatInterface
                                document={selectedDocument}
                                documents={documents}
                                onBack={onBack}
                                onSelectDocument={handleSelectDocument}
                                onFileUpload={onFileUpload}
                                onCitationClick={handleCitationClick}
                                isEmbedded={true}
                            />
                        </div>
                    </Panel>

                    {/* Resize Handle */}
                    <PanelResizeHandle className="w-2 bg-gray-300 hover:bg-gray-400 transition-colors cursor-col-resize flex items-center justify-center">
                        <div className="w-1 h-8 bg-gray-500 rounded-full"></div>
                    </PanelResizeHandle>

                    {/* Right Panel - Document Viewer */}
                    <Panel defaultSize={50} minSize={30}>
                        <div className="h-full">
                            <DocumentViewer
                                document={selectedDocument}
                                highlightPage={highlightPage}
                            />
                        </div>
                    </Panel>
                </PanelGroup>
            </div>
        </div>
    )
}