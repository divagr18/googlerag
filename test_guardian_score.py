#!/usr/bin/env python3
"""
Test script to analyze the shady employment contract with enhanced Guardian Score system
"""
import asyncio
import sys
import os

# Add the parent directory to the path so we can import from api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.core.guardian_score import GuardianScoreAnalyzer

async def test_shady_contract():
    """Test the enhanced Guardian Score system with the problematic employment contract"""
    
    # Read the shady contract
    contract_path = "D:/AgentOS/shady_contract.txt"
    try:
        with open(contract_path, 'r', encoding='utf-8') as f:
            contract_text = f.read()
    except FileNotFoundError:
        print(f"Contract file not found: {contract_path}")
        return
    
    print("🛡️ Testing Enhanced Guardian Score System")
    print("=" * 60)
    print(f"📄 Analyzing contract: {len(contract_text)} characters")
    print()
    
    # Initialize the analyzer
    analyzer = GuardianScoreAnalyzer()
    
    # Analyze the contract (without AI for now to test regex patterns)
    try:
        result = await analyzer.analyze_contract(contract_text)
        
        print(f"🎯 GUARDIAN SCORE: {result.overall_score}/100")
        print(f"⚠️ RISK LEVEL: {result.risk_level.value.upper()}")
        print()
        
        if result.exploitation_flags:
            print(f"🚨 EXPLOITATION FLAGS DETECTED: {len(result.exploitation_flags)}")
            print("-" * 60)
            
            # Group flags by severity for better display
            critical_flags = [f for f in result.exploitation_flags if f.severity_score >= 90]
            high_flags = [f for f in result.exploitation_flags if 80 <= f.severity_score < 90]
            medium_flags = [f for f in result.exploitation_flags if 70 <= f.severity_score < 80]
            low_flags = [f for f in result.exploitation_flags if f.severity_score < 70]
            
            if critical_flags:
                print(f"\n🔴 CRITICAL ISSUES ({len(critical_flags)}):")
                for flag in critical_flags:
                    print(f"   • {flag.description}")
                    print(f"     Type: {flag.type.value}")
                    print(f"     Severity: {flag.severity_score}/100")
                    print(f"     Clause: {flag.clause_text[:100]}...")
                    print(f"     💡 Recommendation: {flag.recommendation}")
                    print()
            
            if high_flags:
                print(f"\n🟡 HIGH RISK ISSUES ({len(high_flags)}):")
                for flag in high_flags:
                    print(f"   • {flag.description}")
                    print(f"     Type: {flag.type.value}")
                    print(f"     Severity: {flag.severity_score}/100")
                    print(f"     Clause: {flag.clause_text[:100]}...")
                    print()
            
            if medium_flags:
                print(f"\n🟠 MEDIUM RISK ISSUES ({len(medium_flags)}):")
                for flag in medium_flags:
                    print(f"   • {flag.description} (Severity: {flag.severity_score})")
            
            if low_flags:
                print(f"\n🟢 LOWER RISK ISSUES ({len(low_flags)}):")
                for flag in low_flags:
                    print(f"   • {flag.description} (Severity: {flag.severity_score})")
                    
        else:
            print("✅ No exploitation patterns detected (this would be concerning for this contract!)")
        
        if result.missing_protections:
            print(f"\n⚠️ MISSING PROTECTIONS ({len(result.missing_protections)}):")
            for protection in result.missing_protections:
                print(f"   • {protection}")
        
        if result.fair_clauses:
            print(f"\n✅ FAIR CLAUSES FOUND ({len(result.fair_clauses)}):")
            for clause in result.fair_clauses:
                print(f"   • {clause}")
        
        print("\n" + "=" * 60)
        print("📊 SUMMARY:")
        print(f"   Guardian Score: {result.overall_score}/100")
        print(f"   Risk Level: {result.risk_level.value}")
        print(f"   Total Issues: {len(result.exploitation_flags)}")
        print(f"   Critical Issues: {len([f for f in result.exploitation_flags if f.severity_score >= 90])}")
        print(f"   Missing Protections: {len(result.missing_protections)}")
        
        if result.overall_score < 30:
            print("\n🚨 RECOMMENDATION: DO NOT SIGN THIS CONTRACT!")
            print("   This contract contains numerous exploitative terms that could cause serious harm.")
            print("   Seek legal counsel immediately before proceeding.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_shady_contract())