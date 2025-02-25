const purgecss = require('@fullhuman/postcss-purgecss');

module.exports = {
  plugins: [
    require('cssnano')({
      preset: [
        'default',
        {
          discardComments: { removeAll: true },
          normalizeWhitespace: true,
          mergeRules: true,
          reduceTransforms: true,
          minifySelectors: true,
        },
      ],
    }),
    purgecss.default({
      content: ['./theme/landing.html'],
    }),
  ],
};