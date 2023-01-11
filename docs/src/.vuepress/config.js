const { description } = require('../../package')

module.exports = {
  /**
   * Ref：https://v1.vuepress.vuejs.org/config/#title
   */
  title: 'TMR4222 Marine Machinery - Viscous Flow in Pipes',
  /**
   * Ref：https://v1.vuepress.vuejs.org/config/#description
   */
  description: description,
  base: '/TMR4222_2023S/',

  /**
   * Extra tags to be injected to the page HTML `<head>`
   *
   * ref：https://v1.vuepress.vuejs.org/config/#head
   */
  head: [
    ['meta', { name: 'theme-color', content: '#3eaf7c' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-status-bar-style', content: 'black' }],
    ['script',
      {
        src: "https://polyfill.io/v3/polyfill.js?features=es6"
      }
    ],
    ['script',
      {
        src: "https://cdn.jsdelivr.net/npm/mathjax@3.0.1/es5/tex-mml-chtml.js"
      }
    ]
  ],

  /**
   * Theme configuration, here is the default theme configuration for VuePress.
   *
   * ref：https://v1.vuepress.vuejs.org/theme/default-theme-config.html
   */
  themeConfig: {
    repo: '',
    editLinks: false,
    docsDir: '',
    editLinkText: '',
    lastUpdated: false,
    sidebar: [
      {
        title: 'Introduction',
        path: '/',
        collapsable: false,
      },
      {
        title: 'Python with Anaconda',
        path: '/python-with-anaconda/',
        collapsable: false
      },
      {
        title: 'Jupyter Notebook',
        path: '/jupyter-notebook/',
        collapsable: false
      },
      {
        title: 'Fluids Package',
        path: '/fluids/',
        collapsable: false
      },
      {
        title: 'Pump Analysis',
        path: '/pump-analysis/',
        collapsable: false
      },
      {
        title: 'Pipe Network Analysis',
        path: '/pipe-network-analysis/',
        collapsable: false
      },
    ]
  },

  /**
   * Apply plugins，ref：https://v1.vuepress.vuejs.org/zh/plugin/
   */
  plugins: [
    '@vuepress/plugin-back-to-top',
    '@vuepress/plugin-medium-zoom',
  ]
}
