// Debug script to test WebSocket upload logic
// Add this to browser console to debug the upload form

console.log('=== WebSocket Upload Debug ===');
console.log('selectedFiles:', typeof selectedFiles !== 'undefined' ? selectedFiles : 'undefined');
console.log('selectedFiles length:', typeof selectedFiles !== 'undefined' ? selectedFiles.length : 'N/A');
console.log('directoryUploadCompleted:', typeof directoryUploadCompleted !== 'undefined' ? directoryUploadCompleted : 'undefined');

// Check if the directory input exists and has files
const directoryInput = document.getElementById('directoryInput');
if (directoryInput) {
    console.log('Directory input found');
    console.log('Directory input files:', directoryInput.files.length);
} else {
    console.log('Directory input NOT found');
}

// Check if the handleSubmit function is properly overridden
console.log('handleSubmit function:', typeof handleSubmit);
console.log('Original handleSubmit:', typeof originalHandleSubmit);

// Test the condition that triggers WebSocket upload
if (typeof selectedFiles !== 'undefined' && typeof directoryUploadCompleted !== 'undefined') {
    const hasDirectoryFiles = selectedFiles && selectedFiles.length > 0;
    console.log('hasDirectoryFiles:', hasDirectoryFiles);
    console.log('Would trigger WebSocket upload:', hasDirectoryFiles && !directoryUploadCompleted);
} else {
    console.log('Cannot test upload condition - variables not defined');
}