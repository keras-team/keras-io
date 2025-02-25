const path = require("path");
const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const { EsbuildPlugin } = require("esbuild-loader");
const ImageMinimizerPlugin = require("image-minimizer-webpack-plugin");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyWebpackPlugin = require("copy-webpack-plugin");

const isProduction = process.env.NODE_ENV === "production";

module.exports = {
  mode: isProduction ? "production" : "development",
  devtool: isProduction ? false : "inline-source-map",
  entry: ["./theme/js/index.js","./theme/css/landing.css"],
  output: {
    path: path.join(__dirname, "bundle"),
    filename: "js/[name].[contenthash].min.js",
    clean: true,
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: "esbuild-loader",
        options: {
          target: "es2021",
          minify: false,
        },
      },
      {
        test: /\.css$/,
        use: [
          MiniCssExtractPlugin.loader,
          {
            loader: "css-loader",
            options: { sourceMap: !isProduction, importLoaders: 1 },
          },
          {
            loader: "postcss-loader",
            options: {
              postcssOptions: {
                config: path.join(__dirname, "postcss.config.js"),
              },
            },
          },
        ],
      },
      {
        test: /\.(png|jpe?g|gif)$/i,
        type: "asset/resource",
        generator: {
          filename: "images/[name][ext]",
        },
      },
      {
        test: /\.webp$/,
        type: "asset/resource",
        generator: {
          filename: "icons/[name][ext]",
        },
      },
    ],
  },
  plugins: [
    new MiniCssExtractPlugin({
      filename: "css/[name].[contenthash].min.css",
    }),
    new HtmlWebpackPlugin({
      template: "./theme/landing.html",
      filename: "landing.html",
      inject: "body",
      minify: isProduction
        ? {
            removeComments: true,
            collapseWhitespace: true,
          }
        : false,
    }),
    new CopyWebpackPlugin({
      patterns: [
        {
          from: path.resolve(__dirname, "theme/img"),
          to: "images",
        },
        {
          from: path.resolve(__dirname, "theme/icons"),
          to: "icons",
        },
      ],
    }),
    new ImageMinimizerPlugin({
      generator: [
        {
          type: "asset",
          implementation: ImageMinimizerPlugin.sharpGenerate,
          options: {
            encodeOptions: {
              webp: { quality: 75 },
            },
          },
        },
      ],
    }),
  ],
  optimization: {
    minimize: isProduction,
    minimizer: [
      new EsbuildPlugin({
        target: "es2021",
      }),
    ],
  },
};
