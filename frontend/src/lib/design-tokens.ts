/**
 * Research Atlas Design Tokens
 *
 * Monochrome palette for the Research Atlas visualization.
 * These values mirror the CSS custom properties in globals.css.
 *
 * Use these for:
 * - Three.js color values (which need hex/rgb)
 * - Dynamic styling in JavaScript
 * - Consistent theming across CSS and JS
 */

export const colors = {
  /** Page background - #FAFAFA */
  background: "#FAFAFA",

  /** Cards, panels, surfaces - #FFFFFF */
  surface: "#FFFFFF",

  /** Primary text - #1A1A1A */
  textPrimary: "#1A1A1A",

  /** Secondary/muted text - #666666 */
  textSecondary: "#666666",

  /** Borders, dividers - #E5E5E5 */
  border: "#E5E5E5",

  /** Accent color for actions - #000000 */
  accent: "#000000",

  /** White for contrast on dark backgrounds */
  white: "#FFFFFF",

  /** Destructive actions */
  destructive: "#DC2626",
} as const;

/**
 * Cluster colors for 3D visualization
 * Grayscale shades to differentiate paper clusters
 */
export const clusterColors = {
  cluster1: "#333333",
  cluster2: "#555555",
  cluster3: "#777777",
  cluster4: "#999999",
  cluster5: "#BBBBBB",
  /** Unclustered/noise points */
  noise: "#DDDDDD",
} as const;

/**
 * Get cluster color by index (cycles through available colors)
 */
export function getClusterColor(clusterId: number): string {
  if (clusterId < 0) return clusterColors.noise;
  const clusterKeys = Object.keys(clusterColors).filter((k) =>
    k.startsWith("cluster")
  );
  const index = clusterId % clusterKeys.length;
  return clusterColors[clusterKeys[index] as keyof typeof clusterColors];
}

/**
 * Convert hex color to Three.js compatible number
 * @example hexToThreeColor("#FF0000") // returns 0xFF0000
 */
export function hexToThreeColor(hex: string): number {
  return parseInt(hex.replace("#", ""), 16);
}

/**
 * 3D Visualization specific constants
 */
export const visualization = {
  /** Default point size for papers */
  pointSize: {
    min: 0.05,
    max: 0.15,
    default: 0.08,
  },

  /** Query point appearance */
  queryPoint: {
    color: colors.accent,
    size: 0.12,
    /** Vertical offset so query point floats above paper centroid */
    verticalOffset: 0.3,
  },

  /** Connection line from query to retrieved papers */
  connectionLine: {
    color: colors.textSecondary,
    opacity: 0.4,
  },

  /** Camera defaults */
  camera: {
    fov: 50,
    near: 0.1,
    far: 1000,
    defaultPosition: [5, 5, 5] as const,
  },

  /** Grid/axis helper */
  grid: {
    color: colors.border,
    opacity: 0.3,
  },
} as const;

/**
 * Spacing scale (matches Tailwind defaults)
 */
export const spacing = {
  xs: "0.25rem", // 4px
  sm: "0.5rem", // 8px
  md: "1rem", // 16px
  lg: "1.5rem", // 24px
  xl: "2rem", // 32px
  "2xl": "3rem", // 48px
} as const;

/**
 * Border radius (matches globals.css --radius)
 */
export const radius = {
  sm: "0.375rem", // 6px
  md: "0.5rem", // 8px
  lg: "0.625rem", // 10px (default)
  xl: "1rem", // 16px
  full: "9999px",
} as const;

export type Colors = typeof colors;
export type ClusterColors = typeof clusterColors;
