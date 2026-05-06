import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { h } from 'vue'
import Annotator from './components/Annotator.vue'
import AnnotationList from './components/AnnotationList.vue'
import AiChat from './components/AiChat.vue'
import CodeMasker from './components/CodeMasker.vue'
import MicroChoice from './components/MicroChoice.vue'
import LcLesson from './components/LcLesson.vue'
import PatternMap from './components/PatternMap.vue'
import SrsReview from './components/SrsReview.vue'

const theme: Theme = {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('AnnotationList', AnnotationList)
    app.component('CodeMasker', CodeMasker)
    app.component('MicroChoice', MicroChoice)
    app.component('LcLesson', LcLesson)
    app.component('PatternMap', PatternMap)
    app.component('SrsReview', SrsReview)
  },
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'layout-bottom': () => [h(Annotator), h(AiChat)],
    })
  },
}

export default theme
