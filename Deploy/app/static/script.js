let currentPrediction = null;

async function predictTitle() {
    const titleInput = document.getElementById('titleInput');
    const title = titleInput.value.trim();
    
    if (!title) {
        alert('請輸入文本');
        titleInput.focus();
        return;
    }
    
    hideResult();
    
    try {
        const response = await fetch(`/api/model/prediction?title=${encodeURIComponent(title)}`);
        
        if (!response.ok) {
            throw new Error(`預測失敗：${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        
        currentPrediction = {
            input_title: title,
            predicted_label: result.label,
            predicted_confidence: result.confidence,
            display_name: result.display_name
        };
        
        showResult(result);
        generateBoardButtons();
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert(`預測失敗：${error.message}`);
    }
}


function showResult(result) {
    const resultBlock = document.getElementById('resultBlock');
    const resultText = document.getElementById('resultText');
    const displayName = result.display_name || result.label;
    const confidencePercent = (result.confidence * 100).toFixed(2);

    resultText.textContent = `${displayName} (${confidencePercent}%)`;
    resultBlock.classList.remove('hidden');
}


function hideResult() {
    const resultBlock = document.getElementById('resultBlock');
    resultBlock.classList.add('hidden');
    
    const feedbackMessage = document.getElementById('feedbackMessage');
    feedbackMessage.classList.add('hidden');
}


function generateBoardButtons() {
    const container = document.getElementById('boardButtons');
    container.innerHTML = ''; 
    
    BOARDS.forEach(board => {
        const button = document.createElement('button');
        button.className = 'board-btn';
        button.textContent = BOARD_DISPLAY_NAMES[board];
        button.onclick = () => submitFeedback(board);
        container.appendChild(button);
    });
}


async function submitFeedback(userLabel) {
    if (!currentPrediction) {
        alert('沒有預測結果');
        return;
    }
    
    const feedbackData = {
        input_title: currentPrediction.input_title,
        predicted_label: currentPrediction.predicted_label,
        predicted_confidence: currentPrediction.predicted_confidence,
        user_label: userLabel
    };
    
    try {
        const response = await fetch('/api/model/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(feedbackData)
        });
        
        if (!response.ok) {
            throw new Error(`提交失敗：${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        const feedbackMessage = document.getElementById('feedbackMessage');
        feedbackMessage.textContent = result.message;
        feedbackMessage.classList.remove('hidden');
        
        setTimeout(() => {
            feedbackMessage.classList.add('hidden');
        }, 3000);
        
    } catch (error) {
        console.error('Feedback error:', error);
        alert(`反饋失敗：${error.message}`);
    }
}


function updateCharCount() {
    const titleInput = document.getElementById('titleInput');
    const charCount = document.getElementById('charCount');
    const currentLength = titleInput.value.length;
    
    charCount.textContent = currentLength;
    
    if (currentLength >= 90) {
        charCount.style.color = '#ff5722';
        charCount.style.fontWeight = 'bold';
    } else {
        charCount.style.color = '#333';
        charCount.style.fontWeight = 'normal';
    }
}


document.addEventListener('DOMContentLoaded', () => {
    const titleInput = document.getElementById('titleInput');
    titleInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            predictTitle();
        }
    });
    titleInput.addEventListener('input', updateCharCount);
    titleInput.focus();
});
