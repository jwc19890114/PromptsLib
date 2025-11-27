pub mod analysis {
    use jieba_rs::Jieba;
    use once_cell::sync::Lazy;
    use serde::{Deserialize, Serialize};
    use std::collections::{HashMap, HashSet};
    use uuid::Uuid;

    static TOKENIZER: Lazy<Jieba> = Lazy::new(Jieba::new);
    static STOPWORDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
        [
            "",
            "\u{7684}",
            "\u{4e86}",
            "\u{548c}",
            "\u{4e0e}",
            "\u{5728}",
            "\u{53ca}",
            "\u{4ee5}\u{53ca}",
            "\u{9700}\u{8981}",
            "\u{6211}\u{4eec}",
            "\u{7528}\u{6237}",
            "\u{8fdb}\u{884c}",
            "\u{5e0c}\u{671b}",
            "\u{8bf7}",
            "\u{4f7f}\u{7528}",
            "\u{8fd9}\u{4e2a}",
            "\u{90a3}\u{4e2a}",
            "\u{8fd9}\u{4e9b}",
            "\u{90a3}\u{4e9b}",
            "\u{4e00}\u{4e0b}",
            "\u{4e00}\u{4e2a}",
            "\u{5982}\u{4f55}",
            "\u{600e}\u{4e48}",
            "\u{5417}",
            "\u{5462}",
            "\u{554a}",
            "\u{54e6}",
            "the",
            "and",
            "or",
            "for",
            "with",
            "into",
            "from",
            "to",
            "of",
            "is",
            "are",
        ]
        .into_iter()
        .collect()
    });
    const TARGET_MARKERS: [&str; 7] = [
        "\u{9762}\u{5411}",
        "\u{9488}\u{5bf9}",
        "\u{7ed9}",
        "\u{4e3a}",
        "\u{9002}\u{5408}",
        "\u{63d0}\u{4f9b}\u{7ed9}",
        "\u{9002}\u{7528}\u{4e8e}",
    ];

    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    pub struct PromptAnalysis {
        pub id: String,
        pub summary: String,
        pub suggested_tags: Vec<String>,
        pub length: usize,
        pub topic: Option<String>,
        pub theme: Option<String>,
        pub role: String,
        pub target_entities: Vec<String>,
    }

    pub fn summarize_prompt(body: &str) -> PromptAnalysis {
        summarize_prompt_with_vocab(body, &[])
    }

    pub fn summarize_prompt_with_vocab(body: &str, vocabulary: &[String]) -> PromptAnalysis {
        let normalized = body.trim();
        let summary = if normalized.is_empty() {
            "\u{8bf7}\u{8f93}\u{5165}\u{6709}\u{6548}\u{7684}\u{63d0}\u{793a}\u{8bcd}\u{4ee5}\u{89e6}\u{53d1}\u{5206}\u{6790}"
                .to_string()
        } else {
            format!(
                "\u{63d0}\u{793a}\u{8bcd}\u{6982}\u{89c8}\u{ff1a}{}",
                &normalized.chars().take(160).collect::<String>()
            )
        };

        let tokens = tokenize(normalized);
        let mut keywords = extract_keywords(&tokens, normalized, vocabulary);
        if keywords.is_empty() {
            keywords.push("general".into());
        }
        let target_entities = extract_targets(&tokens);
        let theme = derive_theme(&keywords, &target_entities, normalized);
        let topic = theme.clone().or_else(|| derive_topic(normalized));
        let role = derive_role(normalized);

        PromptAnalysis {
            id: Uuid::new_v4().to_string(),
            summary,
            suggested_tags: keywords.clone(),
            length: normalized.chars().count(),
            topic,
            theme,
            role,
            target_entities,
        }
    }

    fn tokenize(text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        TOKENIZER
            .cut(text, true)
            .into_iter()
            .flat_map(|token| {
                let owned = token.to_string();
                if owned.is_empty() {
                    return Vec::new();
                }
                if owned.chars().all(|c| c.is_ascii()) {
                    owned
                        .split_whitespace()
                        .map(|t| trim_punctuation(t).to_string())
                        .filter(|t| !t.is_empty())
                        .map(|t| t.to_lowercase())
                        .filter(|t| !is_noise_ascii(t))
                        .collect::<Vec<_>>()
                } else {
                    let cleaned = trim_punctuation(&owned);
                    if cleaned.is_empty() {
                        Vec::new()
                    } else {
                        vec![cleaned.to_string()]
                    }
                }
            })
            .collect()
    }

    fn extract_keywords(tokens: &[String], text: &str, vocabulary: &[String]) -> Vec<String> {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            if !is_meaningful(token) || is_numeric_token(token) {
                continue;
            }
            let normalized = normalize_token(token);
            if normalized.is_empty() || STOPWORDS.contains(normalized.as_str()) {
                continue;
            }
            *freq.entry(normalized).or_insert(0) += 1;
        }

        boost_vocabulary_terms(&mut freq, text, vocabulary);

        let mut ranked: Vec<(String, usize)> = freq.into_iter().collect();
        ranked.sort_by(|(a_token, a_count), (b_token, b_count)| {
            b_count
                .cmp(a_count)
                .then_with(|| b_token.len().cmp(&a_token.len()))
                .then_with(|| a_token.cmp(b_token))
        });

        ranked
            .into_iter()
            .map(|(token, _)| token)
            .filter(|token| token.chars().count() >= 2 || token.len() >= 4)
            .take(8)
            .collect()
    }

    fn boost_vocabulary_terms(freq: &mut HashMap<String, usize>, text: &str, vocabulary: &[String]) {
        if vocabulary.is_empty() || text.is_empty() {
            return;
        }

        let lower_text = text.to_lowercase();
        for term in vocabulary {
            let cleaned = term.trim();
            if cleaned.is_empty() {
                continue;
            }
            let is_ascii = cleaned.chars().all(|c| c.is_ascii());
            let normalized = normalize_token(cleaned);
            let haystack = if is_ascii { lower_text.as_str() } else { text };
            let needle = if is_ascii {
                normalized.as_str()
            } else {
                cleaned
            };
            let count = haystack.match_indices(needle).count();
            if count > 0 {
                *freq.entry(normalized.clone()).or_insert(0) += count * 3;
            }
        }
    }

    fn extract_targets(tokens: &[String]) -> Vec<String> {
        let mut targets = Vec::new();
        for (idx, token) in tokens.iter().enumerate() {
            if let Some(marker) = TARGET_MARKERS.iter().find(|marker| token.contains(*marker)) {
                let tail = token.replacen(marker, "", 1).trim().to_string();
                if is_meaningful_str(&tail) {
                    targets.push(normalize_token(&tail));
                    continue;
                }
                if let Some(next) = tokens.get(idx + 1) {
                    if is_meaningful(next) {
                        targets.push(normalize_token(next));
                        continue;
                    }
                }
            }
        }
        targets.sort();
        targets.dedup();
        targets.into_iter().take(5).collect()
    }

    fn derive_topic(text: &str) -> Option<String> {
        text.lines()
            .map(|line| line.trim())
            .find(|line| !line.is_empty())
            .map(|line| line.chars().take(32).collect())
    }

    fn derive_theme(keywords: &[String], targets: &[String], text: &str) -> Option<String> {
        if !targets.is_empty() {
            return Some(targets.join("、"));
        }
        if let Some(first_keyword) = keywords.iter().find(|token| *token != "general") {
            return Some(first_keyword.clone());
        }
        derive_topic(text)
    }

    fn derive_role(text: &str) -> String {
        let window: String = text.chars().take(200).collect();
        let patterns = [
            "\u{4f5c}\u{4e3a}", // 作为
            "\u{4f60}\u{662f}", // 你是
            "\u{4f60}\u{5c06}", // 你将
            "\u{62c5}\u{4efb}", // 担任
            "\u{626e}\u{6f14}", // 扮演
            "role:",
            "角色",
        ];
        for part in window.split(|c| matches!(c, '\u{ff0c}' | '\u{3002}' | '\u{ff1b}' | '\u{ff1a}' | '.' | ';')) {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            if patterns.iter().any(|p| trimmed.contains(p)) {
                return trimmed.chars().take(48).collect();
            }
        }
        "空".to_string()
    }

    fn is_meaningful(token: &str) -> bool {
        is_meaningful_str(token)
    }

    fn is_meaningful_str(token: &str) -> bool {
        let trimmed = trim_punctuation(token);
        if trimmed.is_empty() {
            return false;
        }
        if trimmed.chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
        trimmed.chars().count() > 1 || trimmed.len() > 3
    }

    fn is_numeric_token(token: &str) -> bool {
        let trimmed = trim_punctuation(token);
        if trimmed.is_empty() {
            return false;
        }
        if trimmed.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
        trimmed.chars().any(|c| c.is_ascii_digit())
    }

    fn is_noise_ascii(token: &str) -> bool {
        token.len() <= 1 || token.chars().all(|c| c.is_ascii_digit())
    }

    fn trim_punctuation(token: &str) -> &str {
        token.trim_matches(|c: char| {
            c.is_ascii_punctuation()
                || matches!(
                    c,
                    '\u{ff0c}'
                        | '\u{3002}'
                        | '\u{ff01}'
                        | '\u{ff1f}'
                        | '\u{3001}'
                        | '\u{ff1b}'
                        | '\u{ff1a}'
                        | '\u{ff08}'
                        | '\u{ff09}'
                        | '\u{3010}'
                        | '\u{3011}'
                )
        })
    }

    fn normalize_token(token: &str) -> String {
        let cleaned = trim_punctuation(token);
        if cleaned.is_empty() {
            return String::new();
        }
        if cleaned.chars().all(|c| c.is_ascii()) {
            cleaned.to_lowercase()
        } else {
            cleaned.to_string()
        }
    }
}

pub mod prompts {
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PromptRecord {
        pub id: String,
        pub title: String,
        pub body: String,
    }

    impl PromptRecord {
        pub fn new(title: impl Into<String>, body: impl Into<String>) -> Self {
            Self {
                id: Uuid::new_v4().to_string(),
                title: title.into(),
                body: body.into(),
            }
        }
    }
}

pub mod storage;
