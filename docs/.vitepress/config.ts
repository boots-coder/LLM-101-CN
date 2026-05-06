import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    base: '/LLM-101-CN/',
    lang: 'zh-CN',
    title: 'LLM 101',
    description: '大模型体系化教程',

    markdown: {
      math: true,
      config: (md) => {
        // Pre-process CodeMasker blocks: encode slot content as base64 prop
        // to prevent VitePress markdown from stripping whitespace/indentation
        const originalParse = md.parse.bind(md)
        md.parse = function (src, env) {
          src = src.replace(
            /<CodeMasker([^>]*)>\n([\s\S]*?)<\/CodeMasker>/g,
            (_, attrs, code) => {
              const trimmed = code.replace(/\n$/, '')
              const b64 = Buffer.from(trimmed, 'utf-8').toString('base64')
              return `<CodeMasker${attrs} :code-base64="'${b64}'" />`
            }
          )
          return originalParse(src, env)
        }
      },
    },

    head: [
      ['link', { rel: 'icon', type: 'image/png', href: '/LLM-101-CN/logo.png' }],
    ],

    themeConfig: {
      logo: '/logo.png',
      siteTitle: 'LLM 101',

      darkModeSwitchLabel: '主题',
      darkModeSwitchTitle: '切换到深色模式',
      lightModeSwitchTitle: '切换到浅色模式',

      search: {
        provider: 'local',
        options: {
          translations: {
            button: {
              buttonText: '搜索文档',
              buttonAriaLabel: '搜索文档',
            },
            modal: {
              noResultsText: '无法找到相关结果',
              resetButtonTitle: '清除查询条件',
              footer: {
                selectText: '选择',
                navigateText: '切换',
                closeText: '关闭',
              },
            },
          },
        },
      },

      nav: [
        { text: '基础知识', link: '/fundamentals/' },
        { text: '模型架构', link: '/architecture/' },
        { text: '训练', link: '/training/' },
        { text: '工程化', link: '/engineering/' },
        { text: '应用', link: '/applications/' },
        { text: '深度剖析', link: '/deep-dives/' },
        { text: '练习', link: '/exercises/' },
        { text: 'LC 速通', link: '/leetcode/' },
        { text: '术语表', link: '/glossary' },
        { text: '我的笔记', link: '/annotations' },
      ],

      sidebar: {
        '/fundamentals/': [
          {
            text: '基础知识',
            items: [
              { text: '章节概览', link: '/fundamentals/' },
              { text: '数学基础', link: '/fundamentals/math' },
              { text: 'Python & ML', link: '/fundamentals/python-ml' },
              { text: '神经网络', link: '/fundamentals/neural-networks' },
              { text: 'NLP 基础', link: '/fundamentals/nlp-basics' },
            ],
          },
        ],
        '/architecture/': [
          {
            text: '模型架构',
            items: [
              { text: '章节概览', link: '/architecture/' },
              { text: 'Transformer', link: '/architecture/transformer' },
              { text: '注意力机制', link: '/architecture/attention' },
              { text: '分词器', link: '/architecture/tokenization' },
              { text: '解码策略', link: '/architecture/decoding' },
              { text: 'GPT 架构', link: '/architecture/gpt' },
              { text: 'Llama 架构', link: '/architecture/llama' },
              { text: 'DeepSeek-V3', link: '/architecture/deepseek' },
              { text: 'Scaling Laws', link: '/architecture/scaling-laws' },
              { text: 'Flow Matching', link: '/architecture/flow-matching' },
            ],
          },
        ],
        '/training/': [
          {
            text: '训练',
            items: [
              { text: '章节概览', link: '/training/' },
              { text: '预训练', link: '/training/pretraining' },
              { text: '继续预训练', link: '/training/continue-pretraining' },
              { text: '数据集构建', link: '/training/datasets' },
              { text: '监督微调', link: '/training/sft' },
              { text: '强化学习基础', link: '/training/rl-basics' },
              { text: '偏好对齐', link: '/training/alignment' },
              { text: '推理模型', link: '/training/reasoning' },
              { text: '知识蒸馏', link: '/training/distillation' },
              { text: 'Agent 强化学习', link: '/training/agent-rl' },
              { text: '对齐进阶', link: '/training/alignment-advanced' },
            ],
          },
        ],
        '/engineering/': [
          {
            text: '工程化',
            items: [
              { text: '章节概览', link: '/engineering/' },
              { text: '推理优化', link: '/engineering/inference' },
              { text: '量化', link: '/engineering/quantization' },
              { text: '模型部署', link: '/engineering/deployment' },
              { text: '模型合并', link: '/engineering/merging' },
              { text: 'LLM 安全', link: '/engineering/safety' },
              { text: '分布式训练', link: '/engineering/distributed' },
              { text: '分布式训练实操', link: '/engineering/distributed-hands-on' },
              { text: '评估', link: '/engineering/evaluation' },
              { text: 'GPU 性能分析', link: '/engineering/profiling' },
              { text: 'Ray 分布式框架', link: '/engineering/ray-framework' },
            ],
          },
        ],
        '/applications/': [
          {
            text: '应用',
            items: [
              { text: '章节概览', link: '/applications/' },
              { text: 'Prompt Engineering', link: '/applications/prompt-engineering' },
              { text: 'RAG', link: '/applications/rag' },
              { text: 'Agent', link: '/applications/agents' },
              { text: 'Agent 框架实战', link: '/applications/agent-frameworks' },
              { text: '多模态', link: '/applications/multimodal' },
              { text: 'Harness 工程', link: '/applications/harness-engineering' },
              { text: '端到端实战', link: '/applications/hands-on-project' },
            ],
          },
        ],
        '/deep-dives/': [
          {
            text: '深度剖析',
            items: [
              { text: '章节概览', link: '/deep-dives/' },
              { text: '手搓 vLLM 推理引擎', link: '/deep-dives/nano-vllm' },
              { text: 'vLLM V1 与 PD 分离架构', link: '/deep-dives/nano-vllm-v1' },
              { text: '深度剖析 GPT-2', link: '/deep-dives/nano-gpt' },
              { text: '深度剖析 LoRA', link: '/deep-dives/lora-from-scratch' },
              { text: '深度剖析 RLHF Pipeline', link: '/deep-dives/nano-rlhf' },
              { text: '手搓 Agent RL', link: '/deep-dives/nano-agent-rl' },
              { text: 'R1 推理模型复现', link: '/deep-dives/r1-reproduction' },
              { text: 'minimind 端到端复现', link: '/deep-dives/minimind-end-to-end' },
              { text: 'Kimi K2 内部机制', link: '/deep-dives/kimi-k2-internals' },
              { text: 'DeepSeek-V4 内部机制', link: '/deep-dives/deepseek-v4-internals' },
              { text: 'MMLU likelihood 评估', link: '/deep-dives/eval-mmlu-likelihood' },
              { text: 'HumanEval pass@k', link: '/deep-dives/eval-humaneval' },
              { text: 'LLM-as-Judge', link: '/deep-dives/eval-llm-judge' },
              { text: 'GCG 对抗后缀攻击', link: '/deep-dives/safety-gcg' },
              { text: 'HarmBench 红队评测', link: '/deep-dives/safety-redteam' },
              { text: 'Safe RLHF（PPO-Lagrangian）', link: '/deep-dives/safety-rlhf' },
              { text: 'datatrove 数据流水线', link: '/deep-dives/data-pipeline-datatrove' },
              { text: 'DCLM 质量分类器', link: '/deep-dives/data-quality-classifier-dclm' },
              { text: 'MinHash / SimHash / Suffix Array 去重', link: '/deep-dives/data-dedup-text' },
              { text: '手撕 nano Agent-RL', link: '/deep-dives/nano-agent-rl' },
              { text: '手撕 LLaVA', link: '/deep-dives/llava-from-scratch' },
            ],
          },
        ],
        '/exercises/': [
          {
            text: '基础架构',
            items: [
              { text: '练习说明', link: '/exercises/' },
              { text: 'Transformer 概念题', link: '/exercises/transformer-quiz' },
              { text: 'Attention 代码填空', link: '/exercises/attention-fill' },
              { text: 'Tokenization 分词填空', link: '/exercises/tokenization-fill' },
              { text: 'GPT 实现挑战', link: '/exercises/gpt-build' },
              { text: 'Llama 实现挑战', link: '/exercises/llama-build' },
              { text: 'MoE 代码填空', link: '/exercises/moe-fill' },
              { text: 'MoE 实现挑战', link: '/exercises/moe-build' },
              { text: 'DeepSeek 架构填空', link: '/exercises/deepseek-arch-fill' },
              { text: 'Flash Attention 填空', link: '/exercises/flash-attn-fill' },
              { text: 'Scaling Laws 填空', link: '/exercises/scaling-laws-fill' },
            ],
          },
          {
            text: '训练 & 微调',
            items: [
              { text: '基础组件实现', link: '/exercises/common-fill' },
              { text: '预训练技术填空', link: '/exercises/pretraining-fill' },
              { text: 'SFT 训练 Pipeline', link: '/exercises/sft-training-fill' },
              { text: 'LoRA 代码填空', link: '/exercises/lora-fill' },
              { text: 'DPO/GRPO 填空', link: '/exercises/dpo-grpo-fill' },
              { text: 'PPO 代码填空', link: '/exercises/ppo-fill' },
              { text: 'Agent-RL 代码填空', link: '/exercises/agent-rl-fill' },
              { text: 'RLHF Pipeline 实现挑战', link: '/exercises/rlhf-build' },
            ],
          },
          {
            text: '工程优化',
            items: [
              { text: '推理优化填空', link: '/exercises/inference-fill' },
              { text: '量化技术填空', link: '/exercises/quantization-fill' },
              { text: '分布式训练填空', link: '/exercises/distributed-fill' },
            ],
          },
          {
            text: '应用 & 评估',
            items: [
              { text: 'Prompt 与评估填空', link: '/exercises/prompt-eval-fill' },
              { text: 'RAG 与 Agent 填空', link: '/exercises/rag-agent-fill' },
              { text: 'RAG 系统实现挑战', link: '/exercises/rag-build' },
              { text: '多模态代码填空', link: '/exercises/multimodal-fill' },
            ],
          },
        ],
        '/leetcode/': [
          {
            text: 'LeetCode Hot100 速通',
            items: [
              { text: '🗺️ 关卡地图', link: '/leetcode/' },
            ],
          },
          {
            text: 'D1 · 哈希 / 双指针 / 滑窗',
            collapsed: false,
            items: [
              { text: '哈希', link: '/leetcode/patterns/hashmap' },
              { text: '双指针', link: '/leetcode/patterns/two-pointer' },
              { text: '滑动窗口', link: '/leetcode/patterns/sliding-window' },
            ],
          },
          {
            text: 'D2 · 二分 / 单调栈 / 堆',
            collapsed: true,
            items: [
              { text: '二分查找', link: '/leetcode/patterns/binary-search' },
              { text: '单调栈', link: '/leetcode/patterns/monotonic-stack' },
              { text: '堆', link: '/leetcode/patterns/heap' },
            ],
          },
          {
            text: 'D3 · 链表 / 树',
            collapsed: true,
            items: [
              { text: '链表', link: '/leetcode/patterns/linked-list' },
              { text: '树 DFS', link: '/leetcode/patterns/tree-dfs' },
              { text: '树 BFS', link: '/leetcode/patterns/tree-bfs' },
            ],
          },
          {
            text: 'D4 · Trie / 回溯 / 图',
            collapsed: true,
            items: [
              { text: 'Trie 前缀树', link: '/leetcode/patterns/trie' },
              { text: '回溯', link: '/leetcode/patterns/backtracking' },
              { text: '图搜索', link: '/leetcode/patterns/graph' },
            ],
          },
          {
            text: 'D5 · DP / 位运算',
            collapsed: true,
            items: [
              { text: '一维 DP', link: '/leetcode/patterns/dp-1d' },
              { text: '二维 DP', link: '/leetcode/patterns/dp-2d' },
              { text: '位运算与前缀和', link: '/leetcode/patterns/bit-prefix' },
            ],
          },
          {
            text: '辅助',
            items: [
              { text: '🐍 Python 速查', link: '/leetcode/python-cheatsheet' },
              { text: '📚 今日复习', link: '/leetcode/srs' },
            ],
          },
        ],
      },

      outline: {
        label: '页面导航',
        level: [2, 3],
      },

      docFooter: {
        prev: '上一页',
        next: '下一页',
      },

      returnToTopLabel: '回到顶部',
      sidebarMenuLabel: '菜单',

      socialLinks: [
        { icon: 'github', link: 'https://github.com/boots-coder/LLM-101-CN' },
      ],
    },

    mermaid: {
      theme: 'default',
    },
  })
)
