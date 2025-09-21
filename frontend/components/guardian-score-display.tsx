"use client"

import React, { useState } from "react"
import { Card } from "./ui/card"
import { Button } from "./ui/button"
import { ChevronDown, ChevronRight, Shield, AlertTriangle, AlertCircle, Info } from "lucide-react"
import type { Document, ExploitationFlag } from "../app/page"

interface GuardianScoreDisplayProps {
    document: Document
}

export function GuardianScoreDisplay({ document }: GuardianScoreDisplayProps) {
    const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({})

    // Don't show if not a contract or no analysis
    if (!document.is_contract || !document.exploitation_flags) {
        return null
    }

    const toggleSection = (section: string) => {
        setExpandedSections(prev => ({
            ...prev,
            [section]: !prev[section]
        }))
    }

    const getScoreColor = (score: number) => {
        if (score >= 70) return "text-green-500"
        if (score >= 40) return "text-yellow-500"
        return "text-red-500"
    }

    const getScoreBgColor = (score: number) => {
        if (score >= 70) return "bg-green-500/10 border-green-500/20"
        if (score >= 40) return "bg-yellow-500/10 border-yellow-500/20"
        return "bg-red-500/10 border-red-500/20"
    }

    const getRiskIcon = (riskLevel: string) => {
        switch (riskLevel) {
            case "critical":
                return <AlertTriangle className="w-4 h-4 text-red-500" />
            case "high":
                return <AlertCircle className="w-4 h-4 text-orange-500" />
            case "medium":
                return <Info className="w-4 h-4 text-yellow-500" />
            case "low":
                return <Shield className="w-4 h-4 text-green-500" />
            default:
                return <Info className="w-4 h-4 text-gray-500" />
        }
    }

    const getRiskColor = (riskLevel: string) => {
        switch (riskLevel) {
            case "critical":
                return "text-red-500 bg-red-500/10"
            case "high":
                return "text-orange-500 bg-orange-500/10"
            case "medium":
                return "text-yellow-500 bg-yellow-500/10"
            case "low":
                return "text-green-500 bg-green-500/10"
            default:
                return "text-gray-500 bg-gray-500/10"
        }
    }

    // Group exploitation flags by type
    const groupedFlags = document.exploitation_flags.reduce((acc, flag) => {
        const type = flag.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
        if (!acc[type]) acc[type] = []
        acc[type].push(flag)
        return acc
    }, {} as Record<string, ExploitationFlag[]>)

    const score = document.guardianScore || 0

    return (
        <div className="space-y-4">
            {/* Main Guardian Score Card */}
            <Card className={`p-6 border ${getScoreBgColor(score)}`}>
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                        <Shield className="w-8 h-8 text-blue-500" />
                        <div>
                            <h3 className="text-lg font-semibold text-white">Guardian Score</h3>
                            <p className="text-sm text-gray-400">
                                {document.contract_type?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} Contract Analysis
                            </p>
                        </div>
                    </div>
                    <div className="text-right">
                        <div className={`text-3xl font-bold ${getScoreColor(score)}`}>
                            {score}/100
                        </div>
                        <div className={`text-sm font-medium px-2 py-1 rounded ${getRiskColor(document.risk_level || '')}`}>
                            {(document.risk_level || '').toUpperCase()} RISK
                        </div>
                    </div>
                </div>

                {/* Summary */}
                {document.analysis_summary && (
                    <div className="bg-gray-900/50 rounded-lg p-4 mb-4">
                        <p className="text-sm text-gray-300">{document.analysis_summary}</p>
                    </div>
                )}

                {/* Contract Classification */}
                <div className="text-sm text-gray-400">
                    <span className="font-medium">Classification Confidence:</span> {Math.round((document.classification_confidence || 0) * 100)}%
                </div>
            </Card>

            {/* Exploitation Flags */}
            {Object.keys(groupedFlags).length > 0 && (
                <Card className="border border-red-500/20 bg-red-500/5">
                    <div className="p-4 border-b border-gray-800">
                        <div className="flex items-center space-x-2">
                            <AlertTriangle className="w-5 h-5 text-red-500" />
                            <h4 className="font-semibold text-white">
                                Exploitation Issues ({document.exploitation_flags.length})
                            </h4>
                        </div>
                    </div>

                    <div className="divide-y divide-gray-800">
                        {Object.entries(groupedFlags).map(([category, flags]) => (
                            <div key={category}>
                                <Button
                                    variant="ghost"
                                    className="w-full justify-between p-4 h-auto text-left hover:bg-gray-800/50"
                                    onClick={() => toggleSection(category)}
                                >
                                    <div className="flex items-center space-x-3">
                                        {getRiskIcon(flags[0].risk_level)}
                                        <div>
                                            <div className="font-medium text-white">{category}</div>
                                            <div className="text-sm text-gray-400">{flags.length} issue{flags.length !== 1 ? 's' : ''}</div>
                                        </div>
                                    </div>
                                    {expandedSections[category] ? (
                                        <ChevronDown className="w-4 h-4 text-gray-400" />
                                    ) : (
                                        <ChevronRight className="w-4 h-4 text-gray-400" />
                                    )}
                                </Button>

                                {expandedSections[category] && (
                                    <div className="px-4 pb-4 space-y-3">
                                        {flags.map((flag, index) => (
                                            <div key={index} className="bg-gray-900/50 rounded-lg p-4 space-y-3">
                                                {/* Flag Header */}
                                                <div className="flex items-start justify-between">
                                                    <div className="flex items-center space-x-2">
                                                        {getRiskIcon(flag.risk_level)}
                                                        <span className={`text-sm font-medium px-2 py-1 rounded ${getRiskColor(flag.risk_level)}`}>
                                                            {flag.risk_level.toUpperCase()}
                                                        </span>
                                                        <span className="text-sm font-medium text-white">
                                                            Severity: {flag.severity_score}/100
                                                        </span>
                                                    </div>
                                                </div>

                                                {/* Description */}
                                                <div>
                                                    <h5 className="text-sm font-medium text-white mb-1">Issue:</h5>
                                                    <p className="text-sm text-gray-300">{flag.description}</p>
                                                </div>

                                                {/* Problematic Clause */}
                                                <div>
                                                    <h5 className="text-sm font-medium text-white mb-1">Problematic Clause:</h5>
                                                    <div className="bg-red-900/20 border border-red-500/30 rounded p-3">
                                                        <p className="text-sm text-red-300 font-mono">"{flag.clause_text}"</p>
                                                    </div>
                                                </div>

                                                {/* AI Recommendation */}
                                                {flag.ai_recommendation && (
                                                    <div>
                                                        <h5 className="text-sm font-medium text-white mb-1">🤖 Guardian's Recommendation:</h5>
                                                        <div className="bg-blue-900/20 border border-blue-500/30 rounded p-3">
                                                            <p className="text-sm text-blue-300">{flag.ai_recommendation}</p>
                                                        </div>
                                                    </div>
                                                )}

                                                {/* Fallback Recommendation */}
                                                {!flag.ai_recommendation && flag.recommendation && (
                                                    <div>
                                                        <h5 className="text-sm font-medium text-white mb-1">💡 Recommendation:</h5>
                                                        <div className="bg-green-900/20 border border-green-500/30 rounded p-3">
                                                            <p className="text-sm text-green-300">{flag.recommendation}</p>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </Card>
            )}

            {/* Action Buttons */}
            <div className="flex space-x-3">
                <Button
                    className="bg-blue-600 hover:bg-blue-700 text-white"
                    onClick={() => window.open('/api/v1/ragsys/ideal-contracts/analyze-contract', '_blank')}
                >
                    Download Full Report
                </Button>
                <Button
                    variant="outline"
                    className="border-gray-600 text-gray-300 hover:bg-gray-800"
                >
                    Get Legal Help
                </Button>
            </div>
        </div>
    )
}