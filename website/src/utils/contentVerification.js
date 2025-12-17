/**
 * Plagiarism checker utility
 * This is a simplified implementation for checking content originality
 * In a real-world scenario, this would connect to plagiarism detection services
 */

// Function to check content for potential plagiarism
export const checkPlagiarism = async (content, title = '') => {
  // In a real implementation, this would call a plagiarism detection API
  // For now, we'll implement basic checks
  
  const checks = {
    // Check for potential issues like excessive quoted text
    hasUncreditedQuotes: checkForUncreditedQuotes(content),
    // Check for content structure similarity (simplified)
    contentStructure: analyzeContentStructure(content),
    // Check for reference compliance
    hasValidReferences: checkReferences(content),
  };
  
  // Return a report
  return {
    id: generateId(),
    title,
    timestamp: new Date().toISOString(),
    checks,
    riskLevel: calculateRiskLevel(checks),
    recommendations: generateRecommendations(checks),
  };
};

// Check for uncredited quotes
const checkForUncreditedQuotes = (content) => {
  // Look for quoted text without proper attribution
  const quotePattern = /"[^"]*"/g;
  const matches = content.match(quotePattern) || [];
  
  // Simplified check - in reality this would need more sophisticated analysis
  return matches.length > 5; // If more than 5 quoted strings, flag for review
};

// Analyze content structure
const analyzeContentStructure = (content) => {
  return {
    wordCount: content.split(/\s+/).length,
    uniqueWords: new Set(content.toLowerCase().match(/\b\w+\b/g)).size,
    sentenceVariety: calculateSentenceVariety(content),
  };
};

// Calculate sentence variety
const calculateSentenceVariety = (content) => {
  const sentences = content.split(/[.!?]+/);
  const lengths = sentences.map(s => s.trim().split(/\s+/).length);
  
  // Calculate standard deviation as measure of variety
  const avg = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const variance = lengths.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / lengths.length;
  const stdDev = Math.sqrt(variance);
  
  return {
    avgLength: avg,
    stdDeviation: stdDev,
    count: lengths.length,
  };
};

// Check for proper references/citations
const checkReferences = (content) => {
  // Look for reference sections or citations
  const referencePatterns = [
    /references/i,
    /bibliography/i,
    /citations/i,
    /source:/i,
    /see also:/i,
    /further reading/i,
    /\[([0-9]+|[^]]+)\]/g, // Numeric or author citations like [1] or [Smith2020]
  ];
  
  return referencePatterns.some(pattern => pattern.test(content));
};

// Calculate overall risk level
const calculateRiskLevel = (checks) => {
  let score = 0;
  if (checks.hasUncreditedQuotes) score += 40;
  if (!checks.hasValidReferences) score += 30;
  
  if (score >= 70) return 'high';
  if (score >= 30) return 'medium';
  return 'low';
};

// Generate recommendations
const generateRecommendations = (checks) => {
  const recommendations = [];
  
  if (checks.hasUncreditedQuotes) {
    recommendations.push("Add proper attribution for quoted content");
  }
  
  if (!checks.hasValidReferences) {
    recommendations.push("Include citations and references for external information");
  }
  
  if (recommendations.length === 0) {
    recommendations.push("Content appears original based on initial checks");
  }
  
  return recommendations;
};

// Generate unique ID
const generateId = () => {
  return 'plagiarism-check-' + Date.now() + '-' + Math.round(Math.random() * 100000);
};

// Content verification function
export const verifyContentOriginality = async (content, title = '') => {
  // Perform the plagiarism check
  const report = await checkPlagiarism(content, title);
  
  // Return verification results
  return {
    original: report.riskLevel === 'low',
    report,
    timestamp: new Date().toISOString(),
  };
};

// Default export
export default checkPlagiarism;