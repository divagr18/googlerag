"use client"

import { useState, useEffect } from "react"
import { DocumentLibrary } from "../components/document-library"
import { TwoColumnChat } from "../components/two-column-chat"
import { ApiClient } from "../lib/api"

export type ExploitationFlag = {
  type: string
  risk_level: string
  description: string
  clause_text: string
  severity_score: number
  recommendation: string
  ai_recommendation: string
}

export type Document = {
  document_id: string
  document_url: string
  document_title: string
  file_type: string
  processed_timestamp: string
  chunk_count: number
  guardianScore?: number // Legacy field for compatibility
  isProcessing?: boolean
  // Guardian Score analysis results (backend format)
  guardian_score?: number
  is_contract?: boolean
  contract_type?: string
  classification_confidence?: number
  risk_level?: string
  exploitation_flags?: ExploitationFlag[]
  analysis_summary?: string
}

const mockDocuments: Document[] = [
  {
    document_id: "1",
    document_url: "uploaded://ApartmentLease_2024.pdf",
    document_title: "ApartmentLease_2024.pdf",
    file_type: "pdf",
    processed_timestamp: "2023-10-26T00:00:00Z",
    chunk_count: 15,
    guardianScore: 68,
  },
  {
    document_id: "2",
    document_url: "uploaded://EmploymentContract_Tech.pdf",
    document_title: "EmploymentContract_Tech.pdf",
    file_type: "pdf",
    processed_timestamp: "2023-10-24T00:00:00Z",
    chunk_count: 23,
    guardianScore: 92,
  },
  {
    document_id: "3",
    document_url: "uploaded://ServiceAgreement_Consulting.pdf",
    document_title: "ServiceAgreement_Consulting.pdf",
    file_type: "pdf",
    processed_timestamp: "2023-10-22T00:00:00Z",
    chunk_count: 8,
    guardianScore: 45,
  },
]

export default function Home() {
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null)
  const [documents, setDocuments] = useState<Document[]>([])
  const [isLoading, setIsLoading] = useState(true)

  // Fetch documents on component mount
  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        setIsLoading(true)
        const api = new ApiClient()
        const response = await api.listDocuments()
        setDocuments(response.documents || [])
      } catch (error) {
        console.error('Failed to fetch documents:', error)
        // Fallback to mock data if API fails
        setDocuments(mockDocuments)
      } finally {
        setIsLoading(false)
      }
    }

    fetchDocuments()
  }, [])

  const handleFileUpload = async (files: FileList) => {
    Array.from(files).forEach(async (file) => {
      const isValidFile =
        file.type === "application/pdf" ||
        file.name.toLowerCase().endsWith(".pdf") ||
        file.type === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
        file.type === "application/msword" ||
        file.name.toLowerCase().endsWith(".docx") ||
        file.name.toLowerCase().endsWith(".doc") ||
        file.type === "text/plain" ||
        file.name.toLowerCase().endsWith(".txt")

      if (isValidFile) {
        // Create a temporary document to show uploading state
        const tempDocument: Document = {
          document_id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
          document_url: `uploading://${file.name}`,
          document_title: file.name,
          file_type: file.name.split('.').pop()?.toLowerCase() || 'unknown',
          processed_timestamp: new Date().toISOString(),
          chunk_count: 0,
          guardianScore: 0,
          isProcessing: true,
        }

        setDocuments((prev) => [tempDocument, ...prev])

        try {
          const api = new ApiClient()
          const response = await api.uploadFile(file)

          // Replace the temporary document with the real one
          setDocuments((prev) =>
            prev.map((doc) =>
              doc.document_id === tempDocument.document_id
                ? {
                  document_id: response.document_id,
                  document_url: `uploaded://${file.name}`,
                  document_title: file.name,
                  file_type: file.name.split('.').pop()?.toLowerCase() || 'unknown',
                  processed_timestamp: new Date().toISOString(),
                  chunk_count: response.total_chunks,
                  // Use actual Guardian Score analysis from backend
                  guardianScore: response.guardian_score || undefined,
                  guardian_score: response.guardian_score || undefined, // Keep backend format too
                  is_contract: response.is_contract || false,
                  contract_type: response.contract_type || undefined,
                  classification_confidence: response.classification_confidence || 0,
                  risk_level: response.risk_level || undefined,
                  exploitation_flags: response.exploitation_flags || undefined,
                  analysis_summary: response.analysis_summary || undefined,
                  isProcessing: false,
                }
                : doc
            )
          )
        } catch (error) {
          console.error('Upload failed:', error)
          // Remove the temporary document on error
          setDocuments((prev) => prev.filter(doc => doc.document_id !== tempDocument.document_id))
          // You might want to show an error notification here
        }
      }
    })
  }

  const handleDeleteDocument = async (documentId: string) => {
    try {
      const api = new ApiClient()
      await api.deleteDocument(documentId)

      // Remove the document from the local state
      setDocuments((prev) => prev.filter(doc => doc.document_id !== documentId))

      console.log('Document deleted successfully')
    } catch (error) {
      console.error('Failed to delete document:', error)
      alert('Failed to delete document. Please try again.')
    }
  }

  if (selectedDocument) {
    return (
      <TwoColumnChat
        initialDocument={selectedDocument}
        documents={documents}
        onBack={() => setSelectedDocument(null)}
        onFileUpload={handleFileUpload}
      />
    )
  }

  return (
    <DocumentLibrary
      documents={documents}
      onSelectDocument={setSelectedDocument}
      onFileUpload={handleFileUpload}
      onDeleteDocument={handleDeleteDocument}
    />
  )
}
