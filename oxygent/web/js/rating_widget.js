// Rating Widget Component
(function($) {
    // 评分组件配置
    const ratingConfig = {
        endpoint: "/api/rate",
        statsEndpoint: "/api/ratings/stats"
    };

    // 创建五星评分组件
    function createRatingWidget(traceId) {
        var container = $('<div class="rating-widget">').data('trace-id', traceId);
        
        // 评分标题
        var title = $('<div class="rating-title">How helpful was this response?</div>');
        
        // 五星评分
        var starsContainer = $('<div class="rating-stars">');
        for (let i = 1; i <= 5; i++) {
            var star = $('<button type="button" class="star" data-score="' + i + '">★</button>');
            star.on('click', function() {
                handleStarClick($(this), container);
            });
            star.on('mouseenter', function() {
                highlightStars($(this));
            });
            star.on('mouseleave', function() {
                resetStars(container);
            });
            starsContainer.append(star);
        }
        
        // 快捷标签按钮组
        var tagsContainer = $('<div class="rating-tags">');
        var tags = [
            "回答准确",
            "步骤清晰",
            "信息过时",
            "遗漏关键点",
            "工具调用错误",
            "超出预期"
        ];
        tags.forEach(function(tag) {
            var tagBtn = $('<button type="button" class="tag-btn">' + tag + '</button>');
            tagBtn.on('click', function() {
                $(this).toggleClass('selected');
            });
            tagsContainer.append(tagBtn);
        });
        
        // 评论输入框
        var commentContainer = $('<div class="rating-comment">');
        var commentTextarea = $('<textarea class="comment-input" placeholder="请输入您的反馈（可选）"></textarea>');
        var submitBtn = $('<button type="button" class="submit-btn">提交评分</button>');
        var statusText = $('<div class="status-text"></div>');
        
        submitBtn.on('click', function() {
            submitRating(container);
        });
        
        commentContainer.append(commentTextarea, submitBtn, statusText);
        
        // 组装组件
        container.append(title, starsContainer, tagsContainer, commentContainer);
        
        // 初始化时检查是否已有评分
        checkExistingRating(traceId, container);
        
        return container;
    }

    // 处理星星点击
    function handleStarClick(star, container) {
        var score = parseInt(star.data('score'));
        container.data('score', score);
        
        // 高亮选中的星星
        highlightStars(star);
        
        // 评分 ≤ 2 分时自动展开评论框
        if (score <= 2) {
            container.find('.rating-comment').show();
        }
    }

    // 高亮星星
    function highlightStars(star) {
        var score = parseInt(star.data('score'));
        var stars = star.parent().find('.star');
        
        stars.each(function(index) {
            if (index < score) {
                $(this).addClass('active');
            } else {
                $(this).removeClass('active');
            }
        });
    }

    // 重置星星
    function resetStars(container) {
        var score = container.data('score');
        var stars = container.find('.star');
        
        stars.each(function(index) {
            if (score && index < score) {
                $(this).addClass('active');
            } else {
                $(this).removeClass('active');
            }
        });
    }

    // 检查是否已有评分
    function checkExistingRating(traceId, container) {
        fetch(ratingConfig.endpoint + '/' + traceId)
            .then(function(response) {
                if (!response.ok) {
                    throw new Error('Request failed');
                }
                return response.json();
            })
            .then(function(data) {
                if (data.has_rating) {
                    // 显示已有评分
                    container.data('score', data.score);
                    resetStars(container);
                    container.find('.status-text').text('已评分: ' + data.score + ' 星');
                    container.find('.submit-btn').prop('disabled', true).text('已提交');
                }
            })
            .catch(function(err) {
                console.error('Check rating error:', err);
            });
    }

    // 提交评分
    function submitRating(container) {
        var traceId = container.data('trace-id');
        var score = container.data('score');
        
        if (!score) {
            container.find('.status-text').text('请先选择评分');
            return;
        }
        
        // 获取选中的标签
        var selectedTags = [];
        container.find('.tag-btn.selected').each(function() {
            selectedTags.push($(this).text());
        });
        
        // 获取评论
        var comment = container.find('.comment-input').val().trim();
        if (selectedTags.length > 0) {
            comment = (comment ? comment + ' | ' : '') + selectedTags.join(', ');
        }
        
        // 显示加载状态
        var submitBtn = container.find('.submit-btn');
        var statusText = container.find('.status-text');
        submitBtn.prop('disabled', true).text('提交中...');
        statusText.text('');
        
        // 提交评分
        fetch(ratingConfig.endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                trace_id: traceId,
                score: score,
                comment: comment
            })
        })
        .then(function(response) {
            if (!response.ok) {
                throw new Error('Request failed');
            }
            return response.json();
        })
        .then(function(data) {
            if (data.status === 'ok') {
                statusText.text('评分提交成功！');
                submitBtn.text('已提交');
            } else {
                statusText.text('提交失败，请重试');
                submitBtn.prop('disabled', false).text('提交评分');
            }
        })
        .catch(function(err) {
            console.error('Submit rating error:', err);
            statusText.text('提交失败，请重试');
            submitBtn.prop('disabled', false).text('提交评分');
        });
    }

    // 附加评分组件到消息
    window.attachRatingWidget = function(li, traceId) {
        if (!traceId || !li || li.length === 0) {
            return;
        }
        if (li.find('.rating-widget').length > 0) {
            return;
        }
        li.append(createRatingWidget(traceId));
    };
})(jQuery);
