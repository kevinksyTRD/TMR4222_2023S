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
        type: "text/javascript",
        src: "https://polyfill.io/v3/polyfill.js?features=es6"
      }
    ],
    ['script',
      {
        type: "text/javascript",
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
        title: 'Numerical method in Python',
        path: '/numerical-method-python/',
        collapsable: false
      },
      {
        title: 'Pipe Friction',
        path: '/pipe-friction/',
        collapsable: false
      },
      {
        title: 'Pipe Network Analysis',
        path: '/pipe-network-analysis/',
        collapsable: false
      },
      {
        title: 'Pump Performance',
        path: '/pump-performance/',
        collapsable: false
      },
      {
        title: 'Pump Control',
        path: '/pump-control/',
        collapsable: false
      },
      {
        title: 'Positive Displacement Pump',
        path: '/positive-displacement-pump/',
        collapsable: false
      }
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
