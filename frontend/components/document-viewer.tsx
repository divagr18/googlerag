"use client"

import React, { useState, useEffect, useRef } from "react"
import { Card } from "./ui/card"
import { FileText, Download } from "lucide-react"
import { Button } from "./ui/button"
import type { Document } from "../app/page"

interface DocumentViewerProps {
    document: Document | null
    highlightPage?: number | string
    onSendMessage?: (message: string) => void
}

export function DocumentViewer({ document, highlightPage, onSendMessage }: DocumentViewerProps) {
    const [textContent, setTextContent] = useState<string>("")
    const [isLoadingText, setIsLoadingText] = useState(false)
    const [hoveredClause, setHoveredClause] = useState<{
        flag: any
        position: { x: number; y: number }
    } | null>(null)
    const hideTimeoutRef = useRef<NodeJS.Timeout | null>(null)
    const [showPDFAsText, setShowPDFAsText] = useState(false)

    // Function to generate explanation for exploitation flags
    const getExplanationForFlag = (flag: any) => {
        const typeExplanations = {
            'unfair_termination': 'This clause allows the landlord to terminate your lease without proper justification or notice.',
            'excessive_fees': 'This involves charges that are unreasonably high or not clearly justified.',
            'liability_shift': 'This clause unfairly shifts responsibility for damages or issues to you as the tenant.',
            'privacy_violation': 'This allows excessive intrusion into your personal space or privacy rights.',
            'deposit_issues': 'This relates to problematic security deposit terms that may violate tenant rights.',
            'maintenance_responsibility': 'This unfairly shifts maintenance obligations that should be the landlord\'s responsibility.',
            'automatic_renewal': 'This clause automatically renews your lease without clear consent or proper notice.',
            'penalty_clauses': 'This involves excessive penalties or fines for minor violations.',
            'entry_rights': 'This gives the landlord excessive rights to enter your rental unit.',
            'rent_increase': 'This allows for unreasonable or improperly notified rent increases.',
            'default': 'This clause contains terms that may be unfavorable to your interests.'
        }

        const explanation = typeExplanations[flag.type as keyof typeof typeExplanations] || typeExplanations.default
        return `${explanation} Risk Level: ${flag.risk_level.toUpperCase()}`
    }

    // Function to highlight risky clauses in text content
    const highlightRiskyClauses = (content: string) => {
        if (!document?.exploitation_flags || document.exploitation_flags.length === 0) {
            return content
        }

        let highlightedContent = content
        const highlightColors = {
            'critical': 'bg-red-900 bg-opacity-50 text-white border-l-2 border-red-500',
            'high': 'bg-orange-900 bg-opacity-50 text-white border-l-2 border-orange-500',
            'medium': 'bg-yellow-900 bg-opacity-50 text-white border-l-2 border-yellow-500',
            'low': 'bg-blue-900 bg-opacity-50 text-white border-l-2 border-blue-500'
        }

        // Sort exploitation flags by clause text length (longest first) to avoid nested replacements
        const sortedFlags = [...document.exploitation_flags]
            .filter(flag => flag.clause_text && flag.clause_text.trim().length > 0)
            .sort((a, b) => b.clause_text.length - a.clause_text.length)

        sortedFlags.forEach((flag, index) => {
            if (flag.clause_text) {
                const clauseText = flag.clause_text.trim()
                const riskLevel = flag.risk_level.toLowerCase()
                const colorClass = highlightColors[riskLevel as keyof typeof highlightColors] || highlightColors.medium

                // Create a unique identifier for this clause
                const clauseId = `clause-${index}`

                // Escape special regex characters but allow for some text variations
                const escapedClause = clauseText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
                const regex = new RegExp(`(${escapedClause})`, 'gi')

                highlightedContent = highlightedContent.replace(regex,
                    `<span id="${clauseId}" 
                        class="inline-block px-1 rounded ${colorClass} cursor-pointer transition-all hover:shadow-lg" 
                        data-flag='${JSON.stringify(flag).replace(/'/g, '&#39;')}'
                        onmouseenter="window.showClauseTooltip(event, '${clauseId}')"
                        onmouseleave="window.hideClauseTooltip()"
                        title="${flag.type.replace('_', ' ')}: ${flag.description}">$1</span>`
                )
            }
        })

        return highlightedContent
    }

    // Function to highlight citation matches in text content
    const highlightCitationMatches = (content: string) => {
        if (!highlightPage || !content) {
            return highlightRiskyClauses(content)
        }

        // For text files, highlightPage might contain a text snippet to highlight
        // This would come from the citation's context
        let highlightedContent = content

        // Try to find and highlight the citation context if it's a text snippet
        if (typeof highlightPage === 'string' && highlightPage.length > 10) {
            const escapedText = highlightPage.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
            const regex = new RegExp(`(${escapedText})`, 'gi')
            highlightedContent = highlightedContent.replace(regex,
                `<span id="citation-highlight" class="bg-blue-600 bg-opacity-40 text-white px-1 rounded border-l-2 border-blue-400">$1</span>`
            )
        }

        return highlightRiskyClauses(highlightedContent)
    }

    // Auto-scroll to citation highlight when component mounts or updates
    useEffect(() => {
        if (textContent && highlightPage) {
            setTimeout(() => {
                const highlightElement = window.document.getElementById('citation-highlight')
                if (highlightElement) {
                    highlightElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
                }
            }, 100)
        }
    }, [textContent, highlightPage])

    // Set up global tooltip functions
    useEffect(() => {
        (window as any).showClauseTooltip = (event: MouseEvent, clauseId: string) => {
            // Clear any existing hide timeout
            if (hideTimeoutRef.current) {
                clearTimeout(hideTimeoutRef.current)
                hideTimeoutRef.current = null
            }

            const element = window.document.getElementById(clauseId)
            if (element) {
                const flagData = element.getAttribute('data-flag')
                if (flagData) {
                    try {
                        const flag = JSON.parse(flagData.replace(/&#39;/g, "'"))
                        const rect = element.getBoundingClientRect()
                        setHoveredClause({
                            flag,
                            position: {
                                x: rect.left + rect.width / 2,
                                y: rect.top - 10
                            }
                        })
                    } catch (e) {
                        console.error('Error parsing flag data:', e)
                    }
                }
            }
        }

        (window as any).hideClauseTooltip = () => {
            // Set a 3-second timeout before hiding the tooltip
            hideTimeoutRef.current = setTimeout(() => {
                setHoveredClause(null)
                hideTimeoutRef.current = null
            }, 3000)
        }

        // Add click-away functionality
        const handleClickAway = (event: MouseEvent) => {
            if (hoveredClause) {
                // Check if the click is not on the tooltip or highlighted clause
                const target = event.target as Element
                if (!target.closest('.tooltip-container') && !target.closest('[id^="clause-"]')) {
                    setHoveredClause(null)
                    if (hideTimeoutRef.current) {
                        clearTimeout(hideTimeoutRef.current)
                        hideTimeoutRef.current = null
                    }
                }
            }
        }

        window.document.addEventListener('click', handleClickAway)

        return () => {
            // Clear timeout on cleanup
            if (hideTimeoutRef.current) {
                clearTimeout(hideTimeoutRef.current)
            }
            window.document.removeEventListener('click', handleClickAway)
            delete (window as any).showClauseTooltip
            delete (window as any).hideClauseTooltip
        }
    }, [hoveredClause])

    // Handle sending explanation to chat
    const handleExplainClause = (flag: any) => {
        if (onSendMessage) {
            const message = `Please explain this contract clause in detail: "${flag.clause_text}". This was flagged as ${flag.type.replace('_', ' ')} with ${flag.risk_level} risk level. The description is: ${flag.description}`
            onSendMessage(message)
            setHoveredClause(null)
        }
    }

    // Define file type checks first
    const isPDF = document?.file_type === 'pdf'
    const isWordDoc = document?.file_type === 'docx' || document?.file_type === 'doc'
    const isTextFile = document?.file_type === 'txt'

    // Fetch text content for text files, docx files, and PDFs when in text mode
    useEffect(() => {
        if (document && document.document_url.startsWith('uploaded://')) {
            const shouldFetchText = isTextFile || isWordDoc || (isPDF && showPDFAsText)

            if (shouldFetchText) {
                const fetchTextContent = async () => {
                    setIsLoadingText(true)
                    try {
                        let url = `http://localhost:8000/api/v1/ragsys/file/${document.document_id}`
                        if (isPDF && showPDFAsText) {
                            // For PDFs, we'll need to add a text extraction endpoint
                            url += '?format=text'
                        }
                        const response = await fetch(url)
                        if (response.ok) {
                            const text = await response.text()
                            setTextContent(text)
                        }
                    } catch (error) {
                        console.error('Failed to fetch text content:', error)
                        setTextContent('Error loading file content')
                    } finally {
                        setIsLoadingText(false)
                    }
                }
                fetchTextContent()
            }
        }
    }, [document?.document_id, isTextFile, isWordDoc, isPDF, showPDFAsText])

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

    return (
        <Card className="h-full flex flex-col">
            {/* Document Header */}
            <div className="p-4 border-b border-gray-700 bg-black rounded-t-lg">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                        <FileText className="w-5 h-5 text-blue-400" />
                        <div>
                            <h3 className="font-semibold text-white truncate max-w-md">
                                {document.document_title}
                            </h3>
                            <p className="text-sm text-gray-300">
                                {document.file_type.toUpperCase()} • {document.chunk_count} chunks
                                {highlightPage && (
                                    <span className="ml-2 px-2 py-1 bg-blue-600 text-white rounded text-xs font-medium">
                                        Page {highlightPage}
                                    </span>
                                )}
                            </p>
                            {/* Risk Legend for highlighted clauses */}
                            {document.exploitation_flags && document.exploitation_flags.length > 0 && (isTextFile || isWordDoc) && (
                                <div className="flex items-center gap-2 mt-2 text-xs">
                                    <span className="text-gray-400">Risk levels:</span>
                                    <div className="flex gap-2">
                                        <div className="flex items-center gap-1">
                                            <div className="w-2 h-2 bg-red-500 rounded"></div>
                                            <span className="text-gray-400">Critical</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <div className="w-2 h-2 bg-orange-500 rounded"></div>
                                            <span className="text-gray-400">High</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <div className="w-2 h-2 bg-yellow-500 rounded"></div>
                                            <span className="text-gray-400">Medium</span>
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <div className="w-2 h-2 bg-blue-500 rounded"></div>
                                            <span className="text-gray-400">Low</span>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                    <div className="flex gap-2">
                        {/* PDF Text View Toggle */}
                        {isPDF && document.exploitation_flags && document.exploitation_flags.length > 0 && (
                            <Button
                                variant="outline"
                                size="sm"
                                className="border-gray-600 text-white hover:bg-gray-800"
                                onClick={() => setShowPDFAsText(!showPDFAsText)}
                            >
                                <FileText className="w-4 h-4 mr-2" />
                                {showPDFAsText ? 'PDF View' : 'Text View'}
                            </Button>
                        )}
                        <Button variant="outline" size="sm" className="border-gray-600 text-white hover:bg-gray-800" onClick={() => {
                            const downloadUrl = document.document_url.startsWith('uploaded://')
                                ? `http://localhost:8000/api/v1/ragsys/file/${document.document_id}`
                                : document.document_url;
                            window.open(downloadUrl, '_blank');
                        }}>
                            <Download className="w-4 h-4 mr-2" />
                            Download
                        </Button>
                    </div>
                </div>
            </div>

            {/* Document Content */}
            <div className="flex-1 overflow-hidden">
                <div className="h-full p-4 overflow-auto">
                    {isPDF ? (
                        showPDFAsText ? (
                            /* PDF Text View for Highlighting */
                            <div className="h-full flex flex-col">
                                {isLoadingText ? (
                                    <div className="h-full flex items-center justify-center bg-white rounded">
                                        <div className="text-center">
                                            <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400 animate-pulse" />
                                            <p className="text-gray-600">Extracting PDF text for highlighting...</p>
                                        </div>
                                    </div>
                                ) : textContent ? (
                                    <div className="h-full bg-white rounded overflow-auto">
                                        <div className="p-4">
                                            <div className="text-xs text-gray-500 mb-4 p-2 bg-blue-50 rounded border">
                                                📄 Text view with highlighting enabled. Use the "PDF View" button to return to the original format.
                                            </div>
                                            <div
                                                className="text-sm text-black whitespace-pre-wrap leading-relaxed"
                                                dangerouslySetInnerHTML={{ __html: highlightCitationMatches(textContent) }}
                                            />
                                        </div>
                                    </div>
                                ) : (
                                    <div className="h-full flex items-center justify-center bg-white rounded">
                                        <div className="text-center">
                                            <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                                            <p className="text-gray-600 mb-2">PDF Text Extraction</p>
                                            <p className="text-sm text-gray-500 mb-4">
                                                Unable to extract text from this PDF for highlighting
                                            </p>
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                onClick={() => setShowPDFAsText(false)}
                                            >
                                                Return to PDF View
                                            </Button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            /* Standard PDF Viewer */
                            <div className="h-full flex">
                                {/* PDF Viewer with Custom Highlight Overlay */}
                                <div className={`${document.exploitation_flags && document.exploitation_flags.length > 0 ? 'flex-1' : 'w-full'} relative`}>
                                    {/* PDF Iframe */}
                                    {document.document_url.startsWith('uploaded://') ? (
                                        <iframe
                                            key={`${document.document_id}-${highlightPage || 'default'}`}
                                            src={`http://localhost:8000/api/v1/ragsys/file/${document.document_id}${highlightPage ? `#page=${highlightPage}` : ''}`}
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

                                    {/* PDF Highlighting Overlay */}
                                    {document.exploitation_flags && document.exploitation_flags.length > 0 && (
                                        <div className="absolute inset-0 pointer-events-none">
                                            <div className="absolute top-4 right-4 bg-black/80 text-white p-3 rounded-lg pointer-events-auto">
                                                <div className="text-sm font-medium mb-2">PDF Highlighting</div>
                                                <div className="text-xs text-gray-300 mb-2">
                                                    {document.exploitation_flags.length} risk{document.exploitation_flags.length !== 1 ? 's' : ''} found
                                                </div>
                                                <div className="text-xs text-gray-400 mb-2">
                                                    View details in the side panel →
                                                </div>
                                                <Button
                                                    size="sm"
                                                    variant="outline"
                                                    className="w-full text-xs border-gray-600 hover:bg-gray-700"
                                                    onClick={() => setShowPDFAsText(true)}
                                                >
                                                    Enable Text Highlighting
                                                </Button>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Risky Clauses Panel for PDFs */}
                                {document.exploitation_flags && document.exploitation_flags.length > 0 && (
                                    <div className="w-80 border-l border-gray-700 bg-gray-900 overflow-y-auto">
                                        <div className="p-4">
                                            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                                                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                                                Flagged Clauses ({document.exploitation_flags.length})
                                            </h3>
                                            <div className="space-y-3">
                                                {document.exploitation_flags.map((flag, index) => {
                                                    const getRiskColor = (level: string) => {
                                                        switch (level.toLowerCase()) {
                                                            case 'critical': return 'bg-red-900 border-red-500 text-red-200'
                                                            case 'high': return 'bg-orange-900 border-orange-500 text-orange-200'
                                                            case 'medium': return 'bg-yellow-900 border-yellow-500 text-yellow-200'
                                                            case 'low': return 'bg-blue-900 border-blue-500 text-blue-200'
                                                            default: return 'bg-gray-900 border-gray-500 text-gray-200'
                                                        }
                                                    }

                                                    return (
                                                        <div
                                                            key={index}
                                                            className={`p-3 rounded border-l-4 ${getRiskColor(flag.risk_level)} cursor-pointer hover:bg-opacity-80 transition-all`}
                                                            onClick={() => onSendMessage && onSendMessage(`Please explain this contract clause in detail: "${flag.clause_text}". This was flagged as ${flag.type.replace('_', ' ')} with ${flag.risk_level} risk level.`)}
                                                        >
                                                            <div className="flex items-start justify-between gap-2 mb-2">
                                                                <span className="text-xs font-medium uppercase tracking-wide">
                                                                    {flag.type.replace('_', ' ')}
                                                                </span>
                                                                <span className="text-xs px-2 py-1 rounded bg-current bg-opacity-20 capitalize">
                                                                    {flag.risk_level}
                                                                </span>
                                                            </div>
                                                            <p className="text-sm mb-2">{flag.description}</p>
                                                            {flag.clause_text && (
                                                                <div className="text-xs italic opacity-80 border-l-2 border-current pl-2 mt-2">
                                                                    "{flag.clause_text}"
                                                                </div>
                                                            )}
                                                            <div className="text-xs mt-2 opacity-60">
                                                                Click to explain in chat
                                                            </div>
                                                        </div>
                                                    )
                                                })}
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )
                    ) : isTextFile ? (
                        <div className="h-full flex flex-col">
                            {/* Text File Viewer */}
                            {isLoadingText ? (
                                <div className="h-full flex items-center justify-center bg-gray-900 rounded">
                                    <div className="text-center">
                                        <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400 animate-pulse" />
                                        <p className="text-gray-400">Loading text content...</p>
                                    </div>
                                </div>
                            ) : textContent ? (
                                <div className="h-full bg-white rounded overflow-auto">
                                    <div className="p-4">
                                        <div
                                            className="text-sm text-black whitespace-pre-wrap font-mono leading-relaxed"
                                            dangerouslySetInnerHTML={{ __html: highlightCitationMatches(textContent) }}
                                        />
                                    </div>
                                </div>
                            ) : (
                                <div className="h-full flex items-center justify-center bg-gray-900 rounded">
                                    <div className="text-center">
                                        <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                                        <p className="text-gray-400 mb-2">Text File</p>
                                        <Button variant="outline" size="sm" onClick={() => {
                                            const downloadUrl = `http://localhost:8000/api/v1/ragsys/file/${document.document_id}`;
                                            window.open(downloadUrl, '_blank');
                                        }}>
                                            <Download className="w-4 h-4 mr-2" />
                                            Download to View
                                        </Button>
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : isWordDoc ? (
                        <div className="h-full flex flex-col">
                            {/* Word Document Viewer */}
                            {isLoadingText ? (
                                <div className="h-full flex items-center justify-center bg-gray-900 rounded">
                                    <div className="text-center">
                                        <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400 animate-pulse" />
                                        <p className="text-gray-400">Loading document content...</p>
                                    </div>
                                </div>
                            ) : textContent ? (
                                <div className="h-full bg-white rounded overflow-auto">
                                    <div className="p-4">
                                        <div
                                            className="text-sm text-black whitespace-pre-wrap leading-relaxed"
                                            dangerouslySetInnerHTML={{ __html: highlightCitationMatches(textContent) }}
                                        />
                                    </div>
                                </div>
                            ) : (
                                <div className="h-full flex items-center justify-center bg-gray-900 rounded">
                                    <div className="text-center">
                                        <FileText className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                                        <p className="text-gray-400 mb-2">Word Document</p>
                                        <p className="text-sm text-gray-500 mb-4">
                                            Preview not available for this document type
                                        </p>
                                        <Button variant="outline" size="sm" onClick={() => {
                                            const downloadUrl = `http://localhost:8000/api/v1/ragsys/file/${document.document_id}`;
                                            window.open(downloadUrl, '_blank');
                                        }}>
                                            <Download className="w-4 h-4 mr-2" />
                                            Download to View
                                        </Button>
                                    </div>
                                </div>
                            )}
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
            </div>

            {/* Tooltip for clause explanations */}
            {hoveredClause && (
                <div
                    className="tooltip-container fixed z-50 bg-gray-800 text-white p-3 rounded-lg shadow-xl border border-gray-600 max-w-sm"
                    style={{
                        left: hoveredClause.position.x - 112, // Center the tooltip
                        top: hoveredClause.position.y - 10,
                        transform: 'translateY(-100%)'
                    }}
                >
                    <div className="text-xs font-semibold text-gray-300 mb-1">
                        {hoveredClause.flag.type.replace('_', ' ').toUpperCase()}
                    </div>
                    <div className="text-sm mb-2 leading-tight">
                        {getExplanationForFlag(hoveredClause.flag)}
                    </div>
                    <div className="text-xs text-gray-400 mb-2">
                        {hoveredClause.flag.description}
                    </div>
                    {onSendMessage && (
                        <Button
                            size="sm"
                            variant="outline"
                            className="w-full h-6 text-xs border-gray-600 hover:bg-gray-700"
                            onClick={() => handleExplainClause(hoveredClause.flag)}
                        >
                            Explain in Chat
                        </Button>
                    )}
                    {/* Tooltip arrow */}
                    <div
                        className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0"
                        style={{
                            borderLeft: '6px solid transparent',
                            borderRight: '6px solid transparent',
                            borderTop: '6px solid #374151'
                        }}
                    />
                </div>
            )}
        </Card>
    )
}