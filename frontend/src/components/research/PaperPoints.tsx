"use client";

import { useRef, useState, useMemo, useEffect, memo } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import {
  getClusterColor,
  hexToThreeColor,
  visualization,
} from "@/lib/design-tokens";
import type { Paper } from "./PapersPanel";

interface PaperPointsProps {
  papers: Paper[];
  selectedPaperId?: string;
  highlightedPaperIds?: string[];
  onPaperClick?: (paperId: string) => void;
  onPaperHover?: (paperId: string | null) => void;
}

/**
 * Renders paper points as spheres in 3D space.
 * Size based on chunk count, color based on cluster.
 */
export function PaperPoints({
  papers,
  selectedPaperId,
  highlightedPaperIds = [],
  onPaperClick,
  onPaperHover,
}: PaperPointsProps) {
  return (
    <group>
      {papers.map((paper) => (
        <PaperPoint
          key={paper.paper_id}
          paper={paper}
          isSelected={paper.paper_id === selectedPaperId}
          isHighlighted={highlightedPaperIds.includes(paper.paper_id)}
          onClick={onPaperClick}
          onHover={onPaperHover}
        />
      ))}
    </group>
  );
}

interface PaperPointProps {
  paper: Paper;
  isSelected: boolean;
  isHighlighted: boolean;
  onClick?: (paperId: string) => void;
  onHover?: (paperId: string | null) => void;
}

const PaperPoint = memo(function PaperPoint({
  paper,
  isSelected,
  isHighlighted,
  onClick,
  onHover,
}: PaperPointProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  // Cached Vector3 for animation to avoid allocations in useFrame
  const targetScaleRef = useRef(new THREE.Vector3(1, 1, 1));

  // Calculate size based on chunk count (normalized)
  const size = useMemo(() => {
    const normalized = Math.min(paper.chunk_count / 100, 1);
    return (
      visualization.pointSize.min +
      normalized * (visualization.pointSize.max - visualization.pointSize.min)
    );
  }, [paper.chunk_count]);

  // Memoize geometry args to prevent recreation
  const sphereArgs = useMemo(
    () => [size, 16, 16] as [number, number, number],
    [size]
  );

  const ringArgs = useMemo(
    () => [size * 1.4, size * 1.6, 32] as [number, number, number],
    [size]
  );

  // Get cluster color
  const baseColor = useMemo(
    () => hexToThreeColor(getClusterColor(paper.cluster_id)),
    [paper.cluster_id]
  );

  // Cleanup cursor on unmount - always reset regardless of hover state
  useEffect(() => {
    return () => {
      document.body.style.cursor = "auto";
    };
  }, []);

  // Animate scale on hover/selection
  useFrame(() => {
    if (!meshRef.current) return;
    const targetScale = hovered || isSelected ? 1.3 : 1;
    targetScaleRef.current.set(targetScale, targetScale, targetScale);
    meshRef.current.scale.lerp(targetScaleRef.current, 0.1);
  });

  const handlePointerOver = () => {
    setHovered(true);
    onHover?.(paper.paper_id);
    document.body.style.cursor = "pointer";
  };

  const handlePointerOut = () => {
    setHovered(false);
    onHover?.(null);
    document.body.style.cursor = "auto";
  };

  const handleClick = () => {
    onClick?.(paper.paper_id);
  };

  return (
    <mesh
      ref={meshRef}
      position={paper.coords}
      onPointerOver={handlePointerOver}
      onPointerOut={handlePointerOut}
      onClick={handleClick}
    >
      <sphereGeometry args={sphereArgs} />
      <meshStandardMaterial
        color={baseColor}
        emissive={isHighlighted ? baseColor : 0x000000}
        emissiveIntensity={isHighlighted ? 0.3 : 0}
        roughness={0.5}
        metalness={0.1}
      />

      {/* Tooltip on hover */}
      {hovered && (
        <Html
          position={[0, size + 0.1, 0]}
          center
          style={{ pointerEvents: "none" }}
        >
          <div className="rounded bg-black/80 px-2 py-1 text-xs whitespace-nowrap text-white shadow-lg">
            <p className="max-w-[200px] truncate font-medium">{paper.title}</p>
            <p className="text-white/70">
              {paper.arxiv_id} · {paper.chunk_count} chunks
            </p>
          </div>
        </Html>
      )}

      {/* Selection ring */}
      {isSelected && (
        <mesh>
          <ringGeometry args={ringArgs} />
          <meshBasicMaterial color={0x000000} side={THREE.DoubleSide} />
        </mesh>
      )}
    </mesh>
  );
});
