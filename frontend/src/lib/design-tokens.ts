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
 * Cool/warm gray split for sophisticated distinction
 */
export const clusterColors = {
  // Cool grays (blue undertone) - Tailwind Slate
  coolDark: "#1E293B",
  coolMedium: "#475569",
  coolLight: "#94A3B8",

  // Warm grays (brown undertone) - Tailwind Stone
  warmDark: "#292524",
  warmMedium: "#57534E",
  warmLight: "#A8A29E",

  /** Unclustered/noise points */
  noise: "#D4D4D4",
} as const;

/**
 * Material presets for 3D spheres
 * Varying roughness and metalness creates visual distinction
 */
export const clusterMaterials = {
  matte: { roughness: 0.9, metalness: 0.0 },
  glossy: { roughness: 0.2, metalness: 0.1 },
  metallic: { roughness: 0.35, metalness: 0.7 },
} as const;

export type ClusterMaterial = keyof typeof clusterMaterials;

/**
 * Combined cluster appearance (color + material)
 */
export interface ClusterAppearance {
  color: string;
  material: { roughness: number; metalness: number };
}

/**
 * Ordered cluster appearances for maximum visual distinction
 * Combines cool/warm colors with matte/glossy/metallic materials
 */
const clusterAppearances: ClusterAppearance[] = [
  // Cluster 0: Cool dark + matte (charcoal stone)
  { color: clusterColors.coolDark, material: clusterMaterials.matte },
  // Cluster 1: Warm medium + glossy (polished clay)
  { color: clusterColors.warmMedium, material: clusterMaterials.glossy },
  // Cluster 2: Cool light + metallic (brushed steel)
  { color: clusterColors.coolLight, material: clusterMaterials.metallic },
  // Cluster 3: Warm dark + glossy (polished obsidian)
  { color: clusterColors.warmDark, material: clusterMaterials.glossy },
  // Cluster 4: Cool medium + metallic (gunmetal)
  { color: clusterColors.coolMedium, material: clusterMaterials.metallic },
  // Cluster 5: Warm light + matte (sandstone)
  { color: clusterColors.warmLight, material: clusterMaterials.matte },
];

/**
 * Get cluster appearance (color + material) by index
 * Cycles through 6 distinct combinations
 */
export function getClusterAppearance(clusterId: number): ClusterAppearance {
  if (clusterId < 0) {
    return { color: clusterColors.noise, material: clusterMaterials.matte };
  }
  return clusterAppearances[clusterId % clusterAppearances.length];
}

/**
 * Get cluster color by index (for 2D UI elements like sidebar dots)
 */
export function getClusterColor(clusterId: number): string {
  return getClusterAppearance(clusterId).color;
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
